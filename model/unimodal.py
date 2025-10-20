import torch
import torch.nn as nn

from model.convnext_utils.dinov3_convnext import ContrastiveReconstructModel

class Unimodal(nn.Module):
    def __init__(
            self,
            representation_names,
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

        self.representation_names = representation_names

        self.autoencoder = ContrastiveReconstructModel(
            in_channels=in_channels,
            backbone_arch=backbone_arch,
            encoder_only=encoder_only,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            patch_size=patch_size,
            proj_dim=proj_dim,
            decoder_out_channels=decoder_out_channels,
        )

    def forward(self, x):
        return self.autoencoder(x)