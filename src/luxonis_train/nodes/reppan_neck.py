"""Implementation of the RepPANNeck module.

Adapted from `YOLOv6: A Single-Stage Object Detection Framework for Industrial
Applications`, available at
https://arxiv.org/pdf/2209.02976.pdf.
It has the balance of feature fusion ability and hardware efficiency.
"""


from typing import Literal

from torch import Tensor, nn
from typeguard import check_type

from luxonis_train.nodes.blocks import RepDownBlock, RepUpBlock
from luxonis_train.utils.general import make_divisible

from .luxonis_node import LuxonisNode


class RepPANNeck(LuxonisNode[list[Tensor], list[Tensor]]):
    def __init__(
        self,
        num_heads: Literal[2, 3, 4] = 3,
        channels_list: list[int] | None = None,
        num_repeats: list[int] | None = None,
        depth_mul: float = 0.33,
        width_mul: float = 0.25,
        **kwargs,
    ):
        """Constructor for the RepPANNeck module.

        Args:
            input_channels_shapes (list): List of output shapes from previous module.
            num_heads (Literal[2,3,4], optional): Number of output heads. Defaults to 3.
              ***Note: Should be same also on head in most cases.***
            channels_list (list, optional): List of number of channels for each block.
              Defaults to [256, 128, 128, 256, 256, 512].
            num_repeats (list, optional): List of number of repeats of RepVGGBlock.
              Defaults to [12, 12, 12, 12].
            depth_mul (float, optional): Depth multiplier. Defaults to 0.33.
            width_mul (float, optional): Width multiplier. Defaults to 0.25.
        """

        super().__init__(**kwargs)

        num_repeats = num_repeats or [12, 12, 12, 12]
        channels_list = channels_list or [256, 128, 128, 256, 256, 512]

        self.num_heads = num_heads

        channels_list = [make_divisible(ch * width_mul, 8) for ch in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]
        channels_list, num_repeats = self._fit_to_num_heads(channels_list, num_repeats)

        self.up_blocks = nn.ModuleList()

        in_channels = check_type(self.in_channels, list[int])[-1]
        out_channels = channels_list[0]
        in_channels_next = check_type(self.in_channels, list[int])[-2]
        curr_num_repeats = num_repeats[0]
        up_out_channel_list = [in_channels]  # used in DownBlocks

        for i in range(1, num_heads):
            curr_up_block = RepUpBlock(
                in_channels=in_channels,
                in_channels_next=in_channels_next,
                out_channels=out_channels,
                num_repeats=curr_num_repeats,
            )
            up_out_channel_list.append(out_channels)
            self.up_blocks.append(curr_up_block)
            if len(self.up_blocks) == (num_heads - 1):
                up_out_channel_list.reverse()
                break

            in_channels = out_channels
            out_channels = channels_list[i]
            in_channels_next = check_type(self.in_channels, list[int])[-1 - (i + 1)]
            curr_num_repeats = num_repeats[i]

        self.down_blocks = nn.ModuleList()
        channels_list_down_blocks = channels_list[(num_heads - 1) :]
        num_repeats_down_blocks = num_repeats[(num_heads - 1) :]

        in_channels = out_channels
        downsample_out_channels = channels_list_down_blocks[0]
        in_channels_next = up_out_channel_list[0]
        out_channels = channels_list_down_blocks[1]
        curr_num_repeats = num_repeats_down_blocks[0]

        for i in range(1, num_heads):
            curr_down_block = RepDownBlock(
                in_channels=in_channels,
                downsample_out_channels=downsample_out_channels,
                in_channels_next=in_channels_next,
                out_channels=out_channels,
                num_repeats=curr_num_repeats,
            )
            self.down_blocks.append(curr_down_block)
            if len(self.down_blocks) == (num_heads - 1):
                break

            in_channels = out_channels
            downsample_out_channels = channels_list_down_blocks[2 * i]
            in_channels_next = up_out_channel_list[i]
            out_channels = channels_list_down_blocks[2 * i + 1]
            curr_num_repeats = num_repeats_down_blocks[i]

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        x0 = inputs[-1]
        up_block_outs = []
        for i, up_block in enumerate(self.up_blocks):
            conv_out, x0 = up_block(x0, inputs[-1 - (i + 1)])
            up_block_outs.append(conv_out)
        up_block_outs.reverse()

        outs = [x0]
        for i, down_block in enumerate(self.down_blocks):
            x0 = down_block(x0, up_block_outs[i])
            outs.append(x0)
        return outs

    def _fit_to_num_heads(
        self, channels_list: list[int], num_repeats: list[int]
    ) -> tuple[list[int], list[int]]:
        """Fits channels_list and num_repeats to num_heads by removing or adding items.

        Also scales the numbers based on offset
        """
        if self.num_heads == 3:
            ...
        elif self.num_heads == 2:
            channels_list = [channels_list[0], channels_list[4], channels_list[5]]
            num_repeats = [num_repeats[0], num_repeats[3]]
        elif self.num_heads == 4:
            channels_list = [
                channels_list[0],
                channels_list[1],
                channels_list[1] // 2,
                channels_list[1] // 2,
                channels_list[1],
                channels_list[2],
                channels_list[3],
                channels_list[4],
                channels_list[5],
            ]
            num_repeats = [
                num_repeats[0],
                num_repeats[1],
                num_repeats[1],
                num_repeats[2],
                num_repeats[2],
                num_repeats[3],
            ]
        else:
            raise ValueError(
                f"Specified number of heads ({self.num_heads}) not supported."
            )

        return channels_list, num_repeats
