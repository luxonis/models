#
# Adapted from: https://arxiv.org/pdf/2209.02976.pdf
#


import torch.nn as nn
from torch import Tensor

from luxonis_train.models.modules.luxonis_module import LuxonisModule
from luxonis_train.models.modules import (
    BlockRepeater,
    RepVGGBlock,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils.general import make_divisible
from luxonis_train.utils.types import ModulePacket


class EfficientRep(LuxonisModule):
    def validate(self, inputs: list[Tensor]) -> None:
        assert len(inputs) == 1

    def __init__(
        self,
        channels_list: list = [64, 128, 256, 512, 1024],
        num_repeats: list = [1, 6, 12, 18, 6],
        in_channels: int = 3,
        depth_mul: float = 0.33,
        width_mul: float = 0.25,
        **kwargs
    ):
        """EfficientRep backbone from `YOLOv6: A Single-Stage Object
        Detection Framework for Industrial Applications`,
        https://arxiv.org/pdf/2209.02976.pdf. It uses RepVGG blocks.

        Args:
            channels_list (list, optional): List of number of channels for each block.
            Defaults to [64, 128, 256, 512, 1024].
            num_repeats (list, optional): List of number of repeats of RepBlock.
            Defaults to [1, 6, 12, 18, 6].
            in_channels (int, optional): Number of input channels,
            should be 3 in most cases . Defaults to 3.
            depth_mul (float, optional): Depth multiplier. Defaults to 0.33.
            width_mul (float, optional): Width multiplier. Defaults to 0.25.
        """
        super().__init__(**kwargs)

        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]

        self.repvgg_encoder = RepVGGBlock(
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
                BlockRepeater(
                    block=RepVGGBlock,
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

    def forward(self, inputs: list[ModulePacket]) -> ModulePacket:
        outputs = []
        x = self.repvgg_encoder(inputs[0]["features"][0])
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return {"features": outputs}
