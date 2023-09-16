#
# Adapted from: https://arxiv.org/pdf/2209.02976.pdf
#


import torch.nn as nn

from luxonis_train.models.backbones.base_backbone import BaseBackbone
from luxonis_train.models.modules import (
    RepVGGBlock,
    RepVGGBlockN,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils.general import make_divisible


class EfficientRep(BaseBackbone):
    def __init__(
        self,
        channels_list: list = [64, 128, 256, 512, 1024],
        num_repeats: list = [1, 6, 12, 18, 6],
        in_channels: int = 3,
        depth_mul: float = 0.33,
        width_mul: float = 0.25,
        **kwargs
    ):
        """EfficientRep backbone from `YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications`,
        https://arxiv.org/pdf/2209.02976.pdf. It uses RepVGG blocks.

        Args:
            channels_list (list, optional): List of number of channels for each block. Defaults to [64, 128, 256, 512, 1024].
            num_repeats (list, optional): List of number of repeats of RepBlock. Defaults to [1, 6, 12, 18, 6].
            in_channels (int, optional): Number of input channels, should be 3 in most cases . Defaults to 3.
            depth_mul (float, optional): Depth multiplier. Defaults to 0.33.
            width_mul (float, optional): Width multiplier. Defaults to 0.25.
        """
        super().__init__(**kwargs)

        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
        )

        self.blocks = nn.ModuleList()
        for i in range(4):
            curr_block = nn.Sequential(
                RepVGGBlock(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i + 1],
                    kernel_size=3,
                    stride=2,
                ),
                RepVGGBlockN(
                    in_channels=channels_list[i + 1],
                    out_channels=channels_list[i + 1],
                    num_blocks=num_repeats[i + 1],
                ),
            )
            self.blocks.append(curr_block)

        self.blocks[-1].append(
            SpatialPyramidPoolingBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5,
            )
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs
