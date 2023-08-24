#
# Source: https://github.com/taveraantonio/BiseNetv1
#

import torch
import torch.nn as nn

from luxonis_train.models.modules import ConvModule
from luxonis_train.models.heads.base_heads import BaseSegmentationHead


class BiSeNetHead(BaseSegmentationHead):
    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        upscale_factor: int = 8,
        is_aux: bool = False,
        **kwargs
    ):
        """BiSeNet segmentation head

        Args:
            n_classes (int): NUmber of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original inpuut shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            upscale_factor (int, optional): Factor used for upscaling input. Defaults to 8.
            is_aux (bool, optional): Either use 256 for intermediate channels or 64. Defaults to False.
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
            **kwargs
        )

        intermediate_channels = 256 if is_aux else 64
        out_channels = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(
            input_channels_shapes[self.attach_index][1], intermediate_channels, 3, 1, 1
        )
        self.conv_1x1 = nn.Conv2d(intermediate_channels, out_channels, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.conv_3x3(x[self.attach_index])
        out = self.conv_1x1(out)
        out = self.upscale(out)
        return out
