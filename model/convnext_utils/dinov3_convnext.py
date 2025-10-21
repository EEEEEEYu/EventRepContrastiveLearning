# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn


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
            feats = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                feats.append(x)  # collect after each stage

            # x is last stage (B, C4, H/32, W/32). feats = [C1@H/4, C2@H/8, C3@H/16, C4@H/32]
            x_pool = x.mean(dim=(-2, -1))  # (B, C)
            x_flat = torch.flatten(x, 2).transpose(1, 2)  # (B, Hf*Wf, C)
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x_flat], dim=1))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],                         # (B, C)
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],  # (B, Hf*Wf, C)
                    "x_prenorm": x,                                          # (B, C4, H/32, W/32)
                    "pyramid": feats,                                        # list of 4 feature maps
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


class DWConvLarge(nn.Module):
    """Depthwise 7x7 (large kernel) + pointwise conv; ConvNeXt-ish."""
    def __init__(self, c, k=7, p=3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=p, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ConvNeXtLiteBlock(nn.Module):
    """ConvNeXt-style MLP block with LayerScale + DropPath (channels_first)."""
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0,3,1,2)  # NHWC -> NCHW
        return shortcut + self.drop_path(x)

class AxialAttention2D(nn.Module):
    """Lightweight axial attention along H and W (no shifting/window grid complexity)."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.h_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.w_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        # x: (B,C,H,W) -> (B*W,H,C) and (B*H,W,C)
        B,C,H,W = x.shape
        # H-axis
        xh = x.permute(0,3,2,1).contiguous().view(B*W, H, C)  # (B*W, H, C)
        xh, _ = self.h_attn(xh, xh, xh, need_weights=False)
        xh = xh.view(B, W, H, C).permute(0,3,2,1).contiguous()  # (B,C,H,W)

        # W-axis
        xw = x.permute(0,2,3,1).contiguous().view(B*H, W, C)  # (B*H, W, C)
        xw, _ = self.w_attn(xw, xw, xw, need_weights=False)
        xw = xw.view(B, H, W, C).permute(0,3,1,2).contiguous()  # (B,C,W,H)->(B,C,H,W) after transpose

        x = 0.5 * (xh + xw)
        return self.proj(x)

class UpFuse(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.in_proj(x)
        s = self.skip_proj(skip)
        x = x + s
        x = self.bn(x)
        return self.act(x)


class StageStack(nn.Module):
    """Stack of ConvNeXtLiteBlocks, optional axial attention at the end."""
    def __init__(self, dim, depth, use_axial_attn=False, num_heads=4,
                 drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, steps=max(depth,1)).tolist()
        self.blocks = nn.Sequential(*[
            ConvNeXtLiteBlock(dim, mlp_ratio=4, drop_path=dpr[i], layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ]) if depth > 0 else nn.Identity()
        self.axial = AxialAttention2D(dim, num_heads=num_heads) if use_axial_attn else nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        x = self.axial(x)
        return x


class UNetPyramidDecoderX(nn.Module):
    """
    Scalable decoder with ConvNeXt-like blocks + optional axial attention.
    Inputs: feats_pyr = [C1@H/4, C2@H/8, C3@H/16, C4@H/32]
    """
    def __init__(
        self,
        in_channels_list: List[int],        # [c1,c2,c3,c4] from backbone
        out_channels: int = 3,
        base_channels: int = 384,           # width scale: 256, 384, 512, ...
        depths: tuple = (2, 2, 2, 2),       # (top_depth, up3_depth, up2_depth, up1_depth)
        use_axial_attn: bool = True,
        num_heads: int = 4,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        assert len(in_channels_list) == 4
        c1, c2, c3, c4 = in_channels_list

        # Top (coarsest) processing @ C4
        self.top_in = nn.Conv2d(c4, base_channels, 1, bias=False)
        self.top = StageStack(
            base_channels, depths[0], use_axial_attn=use_axial_attn, num_heads=num_heads,
            drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value
        )

        # Up: C4->C3->C2->C1 with widths reducing
        c_up3 = base_channels                         # H/16
        c_up2 = max(base_channels // 2, 128)          # H/8
        c_up1 = max(base_channels // 4, 96)           # H/4

        self.fuse3 = UpFuse(c_up3, c3, c_up3)
        self.stage3 = StageStack(
            c_up3, depths[1], use_axial_attn=use_axial_attn, num_heads=num_heads,
            drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value
        )

        self.fuse2 = UpFuse(c_up3, c2, c_up2)
        self.stage2 = StageStack(
            c_up2, depths[2], use_axial_attn=use_axial_attn, num_heads=num_heads,
            drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value
        )

        self.fuse1 = UpFuse(c_up2, c1, c_up1)
        self.stage1 = StageStack(
            c_up1, depths[3], use_axial_attn=use_axial_attn, num_heads=num_heads,
            drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value
        )

        # Head
        self.head = nn.Sequential(
            DWConvLarge(c_up1, k=7, p=3),
            nn.Conv2d(c_up1, out_channels, kernel_size=1)
        )

    def forward(self, feats_pyramid: List[Tensor], target_hw: tuple[int, int]) -> Tensor:
        f1, f2, f3, f4 = feats_pyramid  # C1..C4
        x = self.top_in(f4)
        x = self.top(x)

        x = self.fuse3(x, f3)
        x = self.stage3(x)

        x = self.fuse2(x, f2)
        x = self.stage2(x)

        x = self.fuse1(x, f1)
        x = self.stage1(x)

        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.head(x)


class ContrastiveReconstructModel(nn.Module):
    """
    - Backbone: DINOv3-style ConvNeXt
    - Projection head on global [CLS]-like embedding for contrastive alignment
    - Decoder on spatial feature map for reconstruction
    """
    def __init__(self, 
                 in_channels, 
                 backbone_arch: str = "convnext_base", 
                 encoder_only: bool = False,
                 drop_path_rate: float = 0.1, 
                 layer_scale_init_value: float = 1e-6,
                 patch_size: int = None,
                 proj_dim: int = 128, 
                 decoder_out_channels: int = 3,
                 ):
        super().__init__()
        self.backbone = build_backbone(
            backbone_arch, in_chans=in_channels,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            patch_size=patch_size
        )
        backbone_out_dim = getattr(self.backbone, "embed_dim")
        if backbone_out_dim is None:
            raise ValueError("Backbone must expose 'embed_dim' for projection head.")
        self.encoder_only = encoder_only
        if self.encoder_only:
            self.projection = ProjectionHead(in_dim=backbone_out_dim, proj_dim=proj_dim)
            self.decoder = None
        else:
            self.projection = None
            # NEW: UNet-style decoder with ConvNeXt stage channels
            self.decoder = UNetPyramidDecoderX(
                in_channels_list=self.backbone.embed_dims,
                out_channels=decoder_out_channels,
                base_channels=512,            # try 256/384/512/640 depending on GPU
                depths=(3, 5, 5, 3),          # (top, up3, up2, up1) — increase to 3–5 per stage for stronger
                use_axial_attn=True,          # set False for pure ConvNeXt-style if you want max speed
                num_heads=8,                  # increase with width; 4/6/8 are common
                drop_path_rate=0.2,           # match or exceed backbone dpr for regularization
                layer_scale_init_value=1e-6,
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feats = self.backbone.forward_features(x)  # dict (now includes "pyramid")
        global_emb = feats["x_norm_clstoken"]
        spatial_feats = feats["x_prenorm"]
        pyramid = feats["pyramid"]

        if self.encoder_only:
            z = self.projection(global_emb)
            recon = None
        else:
            z = None
            recon = self.decoder(pyramid, target_hw=(x.shape[-2], x.shape[-1]))

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
    p.add_argument("--encoder_only", action="store_true", help="Square input size H=W.")
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
        encoder_only=args.encoder_only,
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
    print(f"Projection z:   {tuple(proj.shape) if proj is not None else None} (B, {args.proj_dim})")
    print(f"Spatial feats:  {tuple(sf.shape)}  (B, C, Hf, Wf)")
    print(f"Reconstruction: {tuple(recon.shape) if recon is not None else None} (B, {args.decoder_out_ch}, H, W)")
    # quick sanity: reconstruction spatial size matches input
    if not args.encoder_only:
        assert recon.shape[-2:] == (H, W), "Decoder output spatial size must match input image size."
    print("Sanity check passed: reconstruction spatial size matches input.")

    # show downsample factor estimate (just for info)
    ds_factor = model._infer_downsample((H, W), sf.shape[-2:])
    print(f"Estimated downsample factor ~ x{ds_factor}")


if __name__ == "__main__":
    main()
