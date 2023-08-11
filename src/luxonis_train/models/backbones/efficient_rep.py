#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/efficientrep.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#

import torch.nn as nn
import torch

from luxonis_train.models.modules import (
    RepVGGBlock,
    RepVGGBlockN,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils.general import make_divisible


class EfficientRep(nn.Module):
    def __init__(
        self,
        channels_list: list,
        num_repeats: list,
        in_channels: int = 3,
        depth_mul: float = 0.33,
        width_mul: float = 0.25,
        is_4head: bool = False,
    ):
        """EfficientRep backbone, normally used with YoloV6 model.

        Args:
            channels_list (list): List of number of channels for each block
            num_repeats (list): List of number of repeats of RepVGGBlock
            in_channels (int, optional): Number of input channels, should be 3 in most cases . Defaults to 3.
            depth_mul (float, optional): Depth multiplier. Defaults to 0.33.
            width_mul (float, optional): Width multiplier. Defaults to 0.25.
            is_4head (bool, optional): Either build 4 headed architecture or 3 headed one \
                (**Important: Should be same also on neck and head**). Defaults to False.
        """
        super().__init__()

        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]

        self.is_4head = is_4head

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
            if i == 3:
                curr_block.append(
                    SpatialPyramidPoolingBlock(
                        in_channels=channels_list[i + 1],
                        out_channels=channels_list[i + 1],
                        kernel_size=5,
                    )
                )

            self.blocks.append(curr_block)

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        start_idx = 0 if self.is_4head else 1  # idx at which we start saving outputs
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= start_idx:
                outputs.append(x)

        return outputs


if __name__ == "__main__":
    num_repeats = [1, 6, 12, 18, 6]
    depth_mul = 0.33

    channels_list = [64, 128, 256, 512, 1024]
    width_mul = 0.25

    model = EfficientRep(
        in_channels=3,
        channels_list=channels_list,
        num_repeats=num_repeats,
        depth_mul=depth_mul,
        width_mul=width_mul,
        is_4head=False,
    )
    model.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        for out in outs:
            print(out.shape)
