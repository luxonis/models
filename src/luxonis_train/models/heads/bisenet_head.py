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
        c1: int = 256,
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
            c1 (int, optional): Number of input channels. Defaults to 256.
            upscale_factor (int, optional): Factor used for upscaling input. Defaults to 8.
            is_aux (bool, optional): Either use 256 for intermediate channels or 64. Defaults to False.
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        ch = 256 if is_aux else 64
        c2 = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x[self.attach_index]))
        return self.upscale(x)


if __name__ == "__main__":
    from luxonis_train.models.backbones import ContextSpatial

    backbone = ContextSpatial()
    backbone.eval()

    head = BiSeNetHead(n_classes=2)
    head.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        outs = head(outs)
        for out in outs:
            print(out.shape)
