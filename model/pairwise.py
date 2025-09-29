import torch
import torch.nn as nn

from model.convnext_utils.convnext_blocks import ConvNeXtEncoder

class Pairwise(nn.Module):
    def __init__(
            self,
            repr_names,
            in_channels_1,
            in_channels_2,
            latent_channels,
            in_size_1,
            in_size_2,
            latent_size,
            stage_depths,
            base_channels,
            channel_multipliers,
            drop_path_rate,
            use_grn,
        ):
        super.__init__()

        self.repr_names = repr_names

        self.encoder1 = ConvNeXtEncoder(
            in_channels=in_channels_1,
            out_channels=latent_channels,
            in_size=in_size_1,
            target_size=latent_size,
            stage_depths=stage_depths,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            drop_path_rate=drop_path_rate,
            use_grn=use_grn,
            save_skips=False,
        )

        self.encoder2 = ConvNeXtEncoder(
            in_channels=in_channels_2,
            out_channels=latent_channels,
            in_size=in_size_2,
            target_size=latent_size,
            stage_depths=stage_depths,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            drop_path_rate=drop_path_rate,
            use_grn=use_grn,
            save_skips=False,
        )

    def forward(self, repr1, repr2):
        repr1_feature_map, repr1_embedding, _ = self.encoder1(repr1)
        repr2_feature_map, repr2_embedding, _ = self.encoder2(repr2)

        return {
            'repr1_feature_map': repr1_feature_map, 
            'repr1_embedding': repr1_embedding,
            'repr2_feature_map': repr2_feature_map,
            'repr2_embedding': repr2_embedding
        }