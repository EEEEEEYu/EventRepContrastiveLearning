import torch
import torch.nn as nn

from model.convnext_utils.convnext_blocks import ConvNeXtAutoEncoder

class Unimodal(nn.Module):
    def __init__(
            self,
            representation_names,
            height,
            width,
            in_channels,
            latent_size,
            latent_channels,
            stage_depths,
            base_channels,
            channel_multipliers,
            drop_path_rate,
            use_grn,
        ):
        super().__init__()

        self.representation_names = representation_names

        self.autoencoder = ConvNeXtAutoEncoder(
            in_channels=in_channels,
            in_size=(height, width),
            target_size=latent_size,
            latent_channels=latent_channels,
            out_channels=in_channels,
            stage_depths=stage_depths,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            drop_path_rate=drop_path_rate,
            use_grn=use_grn,
        )

    def encoder_forward(self, x):
        return self.autoencoder.encoder_forward(x)

    def forward(self, x):
        latent, x_hat = self.autoencoder(x)
        return {
            'latent': latent,
            'x_hat': x_hat,
        }