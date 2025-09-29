import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm applied on (B, C, H, W) by temporary NHWC view."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)              # BCHW -> BHWC
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)           # BHWC -> BCHW


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # L2 over spatial dims
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class GeMPool2d(nn.Module):
    """
    Generalized Mean Pooling.
    GeM(x) = (1/|Ω| sum_{i in Ω} x_i^p )^(1/p)
    where p is learnable (per-channel or shared).
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable_p: bool = True):
        super().__init__()
        if learnable_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps)
        return F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class PoolingBlock(nn.Module):
    """
    Supports four pooling methods:
    - "avg": average pooling
    - "max": max pooling
    - "conv": convolution-based pooling
    - "gem": generalized mean pooling (global only)

    Args:
        pooling_method: str, one of {"avg", "max", "conv", "gem"}
        pooling_channels: list[int], used in conv mode (out channels per layer).
                          For avg/max/gem this can be dummy list of same length.
        pooling_strides: list[int], stride per pooling layer
        pooling_kernel_size: list[int], kernel size per pooling layer
        in_channels: int, required for conv pooling
        gem_p: float, initial value of p for GeM
        gem_learn_p: bool, whether p is learnable in GeM
    """
    def __init__(self, pooling_method: str,
                 pooling_channels: list,
                 pooling_strides: list,
                 pooling_kernel_size: list,
                 in_channels: int,
                 gem_p: float = 3.0,
                 gem_learn_p: bool = True):
        super().__init__()
        assert pooling_method in ["avg", "max", "conv", "gem"], \
            f"Unsupported pooling_method {pooling_method}"
        assert len(pooling_channels) == len(pooling_strides) == len(pooling_kernel_size), \
            "pooling_channels, strides, and kernel_size must have same length"

        self.method = pooling_method
        self.num_layers = len(pooling_channels)

        layers = []
        current_in = in_channels

        if self.method == "gem":
            # Only one global pooling at the end
            assert self.num_layers == 1, "GeM pooling only supports one global layer"
            layers.append(GeMPool2d(p=gem_p, learnable_p=gem_learn_p))
        else:
            for i in range(self.num_layers):
                stride = pooling_strides[i]
                ksize = pooling_kernel_size[i]

                if self.method == "avg":
                    layers.append(nn.AvgPool2d(kernel_size=ksize, stride=stride))
                elif self.method == "max":
                    layers.append(nn.MaxPool2d(kernel_size=ksize, stride=stride))
                elif self.method == "conv":
                    out_ch = pooling_channels[i]
                    layers.append(nn.Conv2d(current_in, out_ch,
                                            kernel_size=ksize,
                                            stride=stride,
                                            padding=0,
                                            bias=False))
                    layers.append(nn.BatchNorm2d(out_ch))
                    layers.append(nn.GELU())
                    current_in = out_ch

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DropPath(nn.Module):
    """Stochastic depth per sample (from timm)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -----------------------------
# ConvNeXt Block
# -----------------------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dw_kernel=7, mlp_ratio=4.0, layer_scale_init=1e-6,
                 drop_path=0.0, use_grn=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=dw_kernel, padding=dw_kernel//2, groups=dim)
        self.ln = LayerNorm2d(dim)
        hidden = int(mlp_ratio * dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True) \
                     if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.use_grn:
            x = self.grn(x)
        if self.gamma is not None:
            # apply per-channel scale
            x = x * self.gamma.view(1, -1, 1, 1)
        x = shortcut + self.drop_path(x)
        return x



class DownsamplePlanner:
    @staticmethod
    def _factorize_ratio(src: int, dst: int) -> List[int]:
        """
        Factorize the integer ratio src/dst into a sequence of {2, 2, 2, ...} (and optional initial 4).
        We prefer small steps (2x) after an optional 4x stem. Assumes src % dst == 0.
        """
        assert src % dst == 0
        r = src // dst
        steps = []
        if r >= 4 and r % 2 == 0:
            # Prefer a 4x stem once at the beginning if divisible by 4 (ConvNeXt macro-design).
            while r % 4 == 0 and len(steps) == 0:
                steps.append(4)
                r //= 4
        while r > 1:
            assert r % 2 == 0, "Only powers of 2 supported for gentle downsampling."
            steps.append(2)
            r //= 2
        return steps

    """Plan per-axis strides to reach (H', W') from (H, W) with gentle steps (4x once, then 2x)."""
    def __call__(self, H: int, W: int, Ht: int, Wt: int) -> List[Tuple[int, int]]:
        assert H % Ht == 0 and W % Wt == 0, "Target must evenly divide input."
        h_steps = self._factorize_ratio(H, Ht)
        w_steps = self._factorize_ratio(W, Wt)
        # Zip into stage steps; allow asymmetric if needed.
        plan = []
        i = j = 0
        while i < len(h_steps) or j < len(w_steps):
            sh = h_steps[i] if i < len(h_steps) else 1
            sw = w_steps[j] if j < len(w_steps) else 1
            # Keep strides small; if one axis doesn’t need more downsamples, use 1.
            plan.append((sh, sw))
            if i < len(h_steps): i += 1
            if j < len(w_steps): j += 1
        return plan


# -----------------------------
# Encoder
# -----------------------------
class ConvNeXtStage(nn.Module):
    def __init__(self, dim, depth, drop_path_rates, use_grn=False):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                ConvNeXtBlock(dim, drop_path=drop_path_rates[i], use_grn=use_grn)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt-style encoder producing EXACT (B, C_out, Ht, Wt).
    - Gentle downsampling with optional anti-alias blur before each strided conv.
    - Stages: [stem(if needed 4x)] + repeated (LN -> blur -> 2x2 s=2 conv) + ConvNeXt blocks.
    - Returns latent plus skip features for decoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_size: Tuple[int, int],
        target_size: Tuple[int, int],
        stage_depths: List[int] = (3, 3, 9, 3),
        base_channels: int = 32,
        channel_multipliers: List[int] = (1, 2, 4, 8),
        drop_path_rate: float = 0.1,
        use_grn: bool = False,
        save_skips: bool = True,
        pooling_method: str = 'conv',
        pooling_channels: List[int] = [128, 256],
        pooling_kernel_size: List[int| Tuple[int, int]] = [2, 2],
        pooling_strides: List[int| Tuple[int, int]] = [2, 2],
        gem_pooling_p: float = 3.0,
        gem_pooling_learnable_p: bool = True,
        final_latent_embedding_dim: int = 2048,
    ):
        super().__init__()
        H, W = in_size
        Ht, Wt = target_size
        plan = DownsamplePlanner()(H, W, Ht, Wt)

        # Channel schedule across stages (grow as we go down)
        dims = [base_channels * m for m in channel_multipliers]
        total_blocks = sum(stage_depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.save_skips = save_skips
        if self.save_skips:
            self.skips_idx = []  # store skip positions

        # Initial projection to base_channels
        self.in_proj = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)

        # Build stages aligned with the plan
        dpr_ptr = 0
        curr_dim = dims[0]
        for si, (sh, sw) in enumerate(plan):
            # Downsampler for this stage (LN -> blur -> stride conv)
            if sh > 1 or sw > 1:
                self.downsamplers.append(
                    nn.Sequential(
                        LayerNorm2d(curr_dim),
                        nn.Conv2d(curr_dim, dims[min(si, len(dims)-1)], kernel_size= (4 if (sh==4 and sw==4) else 2),
                                  stride=(sh if sh>1 else 1, sw if sw>1 else 1))
                    )
                )
                curr_dim = dims[min(si, len(dims)-1)]
            else:
                # Identity placeholder when no spatial downsample needed
                self.downsamplers.append(nn.Identity())

            depth = stage_depths[min(si, len(stage_depths)-1)]
            drop_slice = dpr[dpr_ptr:dpr_ptr+depth]
            dpr_ptr += depth
            stage = ConvNeXtStage(curr_dim, depth, drop_slice, use_grn=use_grn)
            self.stages.append(stage)
            if self.save_skips:
                self.skips_idx.append(si)

        # Final channel projection to out_channels (latent C')
        self.out_proj = nn.Conv2d(curr_dim, out_channels, kernel_size=1)

        # Global embedding projection
        if pooling_method != 'none':
            self.pooling_block = PoolingBlock(
                pooling_method=pooling_method,
                pooling_channels=pooling_channels,
                pooling_strides=pooling_strides,
                pooling_kernel_size=pooling_kernel_size,
                in_channels=out_channels,
                gem_p=gem_pooling_p,
                gem_learn_p=gem_pooling_learnable_p,
            )

            # We assume stride and kernel size are all 2
            flattened_dim = (target_size[0] // (2 ** len(pooling_strides))) * (target_size[1] // (2 ** len(pooling_strides))) * pooling_channels[-1]

            self.global_embedding_layers = nn.Sequential(
                nn.Linear(flattened_dim, final_latent_embedding_dim),
                nn.ReLU(),
                nn.Linear(final_latent_embedding_dim, final_latent_embedding_dim)
            )
        else:
            self.pooling_block = None
            self.global_embedding_layers = None

    def forward(self, x):
        B, *_ = x.shape
        x = self.in_proj(x)
        if self.save_skips:
            skips = []
            skips.append(x)
        for ds, stage in zip(self.downsamplers, self.stages):
            x = ds(x)
            x = stage(x)
            if self.save_skips:
                skips.append(x)
        x = self.out_proj(x)

        if self.pooling_block is not None:
            e = self.pooling_block(x)
            e = e.reshape(B, -1)
            e = self.global_embedding_layers(e)
        else:
            e = None
        return x, e, skips if self.save_skips else None
        

# -----------------------------
# Decoder (U-Net-ish with ConvNeXt blocks)
# -----------------------------
class ConvNeXtUpBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, depth=2, drop_path=0.0, use_grn=False):
        super().__init__()
        self.proj = nn.Conv2d(in_dim + skip_dim, out_dim, kernel_size=1)
        blocks = []
        for i in range(depth):
            blocks.append(ConvNeXtBlock(out_dim, drop_path=drop_path, use_grn=use_grn))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, skip):
        # upsample by 2 (nearest + conv for smoothing)
        x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.blocks(x)
        return x


class ConvNeXtDecoder(nn.Module):
    def __init__(self, latent_dim: int, skip_dims: List[int], out_channels: int,
                 block_depth=2, use_grn=False):
        super().__init__()
        ups = []
        in_dim = latent_dim
        # use ALL skips (deepest -> shallowest, including the pre-stem)
        for sd in reversed(skip_dims):
            ups.append(ConvNeXtUpBlock(in_dim, sd, sd, depth=block_depth, use_grn=use_grn))
            in_dim = sd
        self.up_blocks = nn.ModuleList(ups)
        self.head = nn.Conv2d(in_dim, out_channels, kernel_size=1)

    def forward(self, latent, skips):
        x = latent
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
        return self.head(x)


# -----------------------------
# Example wrapper
# -----------------------------
class ConvNeXtAutoEncoder(nn.Module):
    """
    Example end-to-end model:
    - Encoder produces exact (B, C_latent, Ht, Wt)
    - Decoder reconstructs to input size (via skips)
    """
    def __init__(self,
                in_channels: int,
                in_size: Tuple[int, int],
                target_size: Tuple[int, int],
                latent_channels: int = 64,
                out_channels: int = None,
                stage_depths: List[int] = (3,3,9,3),
                base_channels: int = 32,
                channel_multipliers: List[int] = (1,2,4,8),
                drop_path_rate: float = 0.1,
                use_grn: bool = True,
                save_skips: bool = True,
                pooling_method: str = 'none',
            ):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            in_size=in_size,
            target_size=target_size,
            stage_depths=stage_depths,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            drop_path_rate=drop_path_rate,
            use_grn=use_grn,
            save_skips=save_skips,
            pooling_method=pooling_method
        )
        # Infer skip dims by running a dummy spec or deterministically from config
        skip_dims = [base_channels * channel_multipliers[0]] + [base_channels * m for m in channel_multipliers][:len(self.encoder.stages)]
        self.decoder = ConvNeXtDecoder(
            latent_dim=latent_channels,
            skip_dims=skip_dims,
            out_channels=out_channels if out_channels is not None else in_channels,
            block_depth=2,
            use_grn=use_grn
        )

    def forward(self, x):
        latent, _, skips = self.encoder(x)
        recon = self.decoder(latent, skips)
        return latent, recon
    

def autoencoder_test():
    model = ConvNeXtAutoEncoder(
        in_channels=10,
        in_size=(640, 480),
        target_size=(20, 15),
        latent_channels=64,
        out_channels=10,
        stage_depths=(3,3,9,3),
        base_channels=32,
        channel_multipliers=(1,2,4,8),
        drop_path_rate=0.1,
        use_grn = True,
    )

    B, C, H, W = 5, 10, 640, 480
    test_input = torch.randn((B, C, H, W))
    latent, recon = model(test_input)
    print(f"latent.shape: {latent.shape} recon.shape: {recon.shape}")

def encoder_test():
    encoder = ConvNeXtEncoder(
        in_channels=10,
        out_channels=64,
        in_size=[640, 480],
        target_size=[20, 15],
        stage_depths=(3, 3, 9, 3),
        base_channels=32,
        channel_multipliers=[1, 2, 4, 8],
        drop_path_rate=0.1,
        use_grn=0.1,
        save_skips=False,
        pooling_method='conv',
        pooling_channels=[128, 256],
        pooling_kernel_size=[2, 2],
        pooling_strides=[2, 2],
        gem_pooling_p=3.0,
        gem_pooling_learnable_p=True,
        final_latent_embedding_dim=2048,
    )

    B, C, H, W = 5, 10, 640, 480
    test_input = torch.randn((B, C, H, W))
    latent, embedding, _ = encoder(test_input)
    print(f"latent.shape: {latent.shape} global_embedding shape: {embedding.shape}")


if __name__ == '__main__':
    # autoencoder_test()
    encoder_test()