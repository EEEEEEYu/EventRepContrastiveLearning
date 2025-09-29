import torch
import torch.nn as nn

from model.convnext_utils.convnext_blocks import ConvNeXtAutoEncoder

class Unimodal(nn.Module):
    def __init__(
            self,
            repr_names,
            in_size,
            in_channels,
            latent_size,
            latent_channels,
            stage_depths,
            base_channels,
            channel_multipliers,
            drop_path_rate,
            use_grn,
        ):
        super.__init__()

        self.repr_names = repr_names

        self.autoencoder = ConvNeXtAutoEncoder(
            in_channels=in_channels,
            in_size=in_size,
            target_size=latent_size,
            latent_channels=latent_channels,
            out_channels=in_channels,
            stage_depths=stage_depths,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            drop_path_rate=drop_path_rate,
            use_grn=use_grn,
        )

    def forward(self, x):
        latent, x_hat = self.autoencoder(x)
        return latent, x_hat