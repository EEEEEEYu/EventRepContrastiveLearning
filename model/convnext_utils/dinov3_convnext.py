# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn


logger = logging.getLogger("dinov3")
logging.basicConfig(level=logging.INFO)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block (channels_last MLP-style implementation)."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm supporting channels_last or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""
    ConvNeXt backbone (DINOv3-style: no classifier head, returns both global & spatial features)

    Args:
        in_chans (int): input channels
        depths (list[int]): blocks per stage
        dims (list[int]): channels per stage
        drop_path_rate (float)
        layer_scale_init_value (float)
        patch_size (int|None): if set, patch tokens can be resized to ViT-like grid
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        patch_size: int | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # stem and 3 downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # stages
        self.stages = nn.ModuleList()
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # final norm (used as DINOv3-style projector norm, not classifier)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # DINOv3 heads/metadata
        self.head = nn.Identity()
        self.embed_dim = dims[-1]
        self.embed_dims = dims
        self.n_blocks = len(self.downsample_layers)
        self.chunked_blocks = False
        self.n_storage_tokens = 0
        self.norms = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.norms.append(self.norm)

        self.patch_size = patch_size
        self.input_pad_size = 4  # initial stride

        # init
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    # ----- DINOv3-style feature API -----
    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            # x is (B, C, Hf, Wf)
            x_pool = x.mean(dim=(-2, -1))  # (B, C)
            x_flat = torch.flatten(x, 2).transpose(1, 2)  # (B, Hf*Wf, C)
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x_flat], dim=1))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],                         # (B, C)
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],  # (B, Hf*Wf, C)
                    "x_prenorm": x,                                          # (B, C, Hf, Wf)
                    "masks": masks,
                }
            )
        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    # helpers for intermediate layers (unused in this script, kept for completeness)
    def _get_intermediate_layers(self, x, n=1):
        h, w = x.shape[-2:]
        output, total_block_len = [], len(self.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = x.mean([-2, -1])
                x_patches = x
                if self.patch_size is not None:
                    x_patches = nn.functional.interpolate(
                        x, size=(h // self.patch_size, w // self.patch_size), mode="bilinear", antialias=True
                    )
                output.append([x_pool, x_patches])
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers(x, n)
        if norm:
            nchw_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                (
                    norm(cls_token),
                    norm(patches.flatten(-2, -1).permute(0, 2, 1)),
                )
                for (cls_token, patches), norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.permute(0, 2, 1).reshape(*nchw).contiguous())
                    for (cls_token, patches), nchw in zip(outputs, nchw_shapes)
                ]
        elif not reshape:
            outputs = [(cls_token, patches.flatten(-2, -1).permute(0, 2, 1)) for (cls_token, patches) in outputs]
        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)




def get_convnext_arch(arch_name: str):
    """
    arch_name format: 'convnext_tiny' | 'convnext_small' | 'convnext_base' | 'convnext_large'
    """
    convnext_sizes = {
        "tiny": dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
        "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
        "base": dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
        "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    }
    try:
        query_sizename = arch_name.split("_")[1]
        size_dict = convnext_sizes[query_sizename]
    except Exception as e:
        raise NotImplementedError(f"Unrecognized ConvNeXt size in '{arch_name}'.") from e

    return partial(ConvNeXt, **size_dict)


class ProjectionHead(nn.Module):
    """
    Projects global embedding into a smaller unified embedding space for contrastive loss.
    """
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        # x: (B, in_dim)
        return F.normalize(self.net(x), p=2, dim=1)


class FlexibleDecoder(nn.Module):
    """
    Decode spatial feature map back to original image size (B, C_out, H, W).
    Works for arbitrary input sizes by:
      - repeated 2x upsampling with convs
      - a final interpolate to the exact target size
    """
    def __init__(self, in_channels: int, out_channels: int = 3, hidden_channels: int = 256, max_upsamples: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Build a stack of upsample+conv blocks; we can stop early in forward
        up_blocks = []
        for _ in range(max_upsamples):
            up_blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.up = nn.Sequential(*up_blocks)
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.max_upsamples = max_upsamples

    def forward(self, x: Tensor, target_hw: tuple[int, int]) -> Tensor:
        """
        x: (B, C_in, Hf, Wf)
        target_hw: (H, W) desired output spatial size (usually original image)
        """
        B, C, Hf, Wf = x.shape
        Ht, Wt = target_hw
        y = self.stem(x)

        # figure out how many 2x steps we need (ceil to avoid under-upscaling)
        scale_h = max(Ht / max(Hf, 1), 1.0)
        scale_w = max(Wt / max(Wf, 1), 1.0)
        steps = int(max(math.ceil(math.log2(scale_h)), math.ceil(math.log2(scale_w))))
        steps = min(max(steps, 0), self.max_upsamples)

        if steps > 0:
            # run exactly 'steps' upsample blocks
            # each step uses 3 modules inside self.up: Upsample, Conv2d, ReLU
            modules_per_step = 3
            up_to = steps * modules_per_step
            y = self.up[:up_to](y)

        # final precise resize + conv head
        if (y.shape[-2], y.shape[-1]) != (Ht, Wt):
            y = F.interpolate(y, size=(Ht, Wt), mode="bilinear", align_corners=False)
        out = self.head(y)
        return out


class ContrastiveReconstructModel(nn.Module):
    """
    - Backbone: DINOv3-style ConvNeXt
    - Projection head on global [CLS]-like embedding for contrastive alignment
    - Decoder on spatial feature map for reconstruction
    """
    def __init__(self, 
                 backbone_arch, 
                 in_channels, 
                 drop_path_rate, 
                 layer_scale_init_value: float = 1e-6,
                 patch_size: int = None,
                 proj_dim: int = 128, 
                 decoder_out_channels: int = 3,
                ):
        super().__init__()
        self.backbone = build_backbone(backbone_arch, in_chans=in_channels, drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value, patch_size=patch_size)
        backbone_out_dim = getattr(self.backbone, "embed_dim")
        if backbone_out_dim is None:
            raise ValueError("Backbone must expose 'embed_dim' for projection head.")
        self.projection = ProjectionHead(in_dim=backbone_out_dim, proj_dim=proj_dim)
        self.decoder = FlexibleDecoder(in_channels=backbone_out_dim, out_channels=decoder_out_channels)

    @torch.no_grad()
    def _infer_downsample(self, x_hw: tuple[int, int], feat_hw: tuple[int, int]) -> int:
        # Useful sanity check if you want to inspect the pyramid factor
        h, w = x_hw
        hf, wf = feat_hw
        if hf == 0 or wf == 0:
            return 0
        return int(round((h / hf + w / wf) * 0.5))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # x : (B, C, H, W)
        feats = self.backbone.forward_features(x)  # dict
        global_emb = feats["x_norm_clstoken"]      # (B, C)
        spatial_feats = feats["x_prenorm"]         # (B, C, Hf, Wf)

        z = self.projection(global_emb)            # (B, proj_dim)
        recon = self.decoder(spatial_feats, target_hw=(x.shape[-2], x.shape[-1]))  # (B, outC, H, W)

        return {"proj": z, "recon": recon, "global_emb": global_emb, "spatial_feats": spatial_feats}


# ---------- Utilities to build backbone & (optionally) load pretrained ----------
def build_backbone(arch: str, in_chans: int = 3, drop_path_rate: float = 0.0, layer_scale_init_value: float = 1e-6, patch_size: int = None) -> nn.Module:
    ctor = get_convnext_arch(arch)
    backbone = ctor(in_chans=in_chans, drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value, patch_size=patch_size)
    return backbone


# ---------- Self-contained testing main ----------
def parse_args():
    p = argparse.ArgumentParser(description="ContrastiveReconstructModel with DINOv3 ConvNeXt backbone")
    p.add_argument("--backbone_arch", type=str, default="convnext_base",
                   choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
                   help="Backbone size.")
    p.add_argument("--image-size", type=int, default=224, help="Square input size H=W.")
    p.add_argument("--batch", type=int, default=4, help="Batch size for the dummy run.")
    p.add_argument("--proj-dim", type=int, default=128, help="Projection head output dim.")
    p.add_argument("--drop-path-rate", type=float, default=0.1, help="Stochastic depth rate.")
    p.add_argument("--layer_scale_init_value", type=float, default=1e-6, help="layer_scale_init_value")
    p.add_argument("--patch_size", type=int, default=16, help="Patch size")
    p.add_argument("--decoder-out-ch", type=int, default=3, help="Decoder output channels.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--in_channels", type=int, default=3, help="Input channels.")
    return p.parse_args()

def print_model_size(model):
    # Calculate the size of the model in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_bytes = param_size + buffer_size
    size_all_mb = size_all_bytes / (1024**2)

    print(f"Model size: {size_all_mb:.3f} MB")

def main():
    args = parse_args()
    torch.manual_seed(0)

    # Wrap into ContrastiveReconstructModel
    model = ContrastiveReconstructModel(
        backbone_arch=args.backbone_arch, 
        in_channels=args.in_channels, 
        drop_path_rate=args.drop_path_rate,
        proj_dim=args.proj_dim,
        decoder_out_channels=args.decoder_out_ch,
        layer_scale_init_value=args.layer_scale_init_value,
        patch_size=args.patch_size,
    ).to(args.device)
    model.eval()

    # Dummy input
    H = W = args.image_size
    x = torch.randn(args.batch, args.in_channels, H, W, device=args.device)

    with torch.no_grad():
        out = model(x)

    proj, recon, g, sf = out["proj"], out["recon"], out["global_emb"], out["spatial_feats"]

    print_model_size(model)

    # Print shapes
    print("=== Test Run ===")
    print(f"Backbone: {args.backbone_arch} | embed_dim={model.backbone.embed_dim}")
    print(f"Input:          {tuple(x.shape)}")
    print(f"Global emb:     {tuple(g.shape)}   (B, C)")
    print(f"Projection z:   {tuple(proj.shape)} (B, {args.proj_dim})")
    print(f"Spatial feats:  {tuple(sf.shape)}  (B, C, Hf, Wf)")
    print(f"Reconstruction: {tuple(recon.shape)} (B, {args.decoder_out_ch}, H, W)")
    # quick sanity: reconstruction spatial size matches input
    assert recon.shape[-2:] == (H, W), "Decoder output spatial size must match input image size."
    print("Sanity check passed: reconstruction spatial size matches input.")

    # show downsample factor estimate (just for info)
    ds_factor = model._infer_downsample((H, W), sf.shape[-2:])
    print(f"Estimated downsample factor ~ x{ds_factor}")


if __name__ == "__main__":
    main()
