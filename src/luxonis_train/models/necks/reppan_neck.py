#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#


import torch
import torch.nn as nn
from typing import Optional, Literal

from luxonis_train.models.necks.base_neck import BaseNeck
from luxonis_train.models.modules import RepBlock, ConvModule
from luxonis_train.utils.general import make_divisible


class RepPANNeck(BaseNeck):
    def __init__(
        self,
        prev_out_shapes: list,
        num_heads: Optional[Literal[2, 3, 4]] = 3,
        channels_list: Optional[list] = [256, 128, 128, 256, 256, 512],
        num_repeats: Optional[list] = [12, 12, 12, 12],
        depth_mul: Optional[float] = 0.33,
        width_mul: Optional[float] = 0.25,
        **kwargs,
    ):
        """RepPANNeck normally used with YoloV6 model. It has the balance of feature fusion ability and hardware efficiency.

        Args:
            prev_out_shapes (list): List of shapes of previous outputs
            num_heads (Literal[2,3,4], optional): Number of output heads.
                (**Important: Should be same also on backbone and head**). Defaults to 3.
            channels_list (list, optional): List of number of channels for each block. Defaults to [256, 128, 128, 256, 256, 512].
            num_repeats (list, optiona): List of number of repeats of RepBlock. Defaults to [12, 12, 12, 12].
            depth_mul (float, optional): Depth multiplier. Defaults to 0.33.
            width_mul (float, optional): Width multiplier. Defaults to 0.25.
        """
        super().__init__(prev_out_shapes=prev_out_shapes)
        if num_heads not in [2, 3, 4]:
            raise ValueError(
                f"Specified number of heads not supported. Choose one of [2,3,4]"
            )

        self.num_heads = num_heads
        # channels_list: [out UpBlock0, out UpBlock1, out downsample0, out DownBlock0, out downsample1, out DownBlock1]
        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        # num_repeats: [UpBlock0, UpBlock1, DownBlock0, DownBlock1]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]
        channels_list, num_repeats = self._fit_to_num_heads(channels_list, num_repeats)

        # create num_heads-1 UpBlocks
        self.up_blocks = nn.ModuleList()

        in_channels = prev_out_shapes[-1][1]
        out_channels = channels_list[0]
        in_channels_next = prev_out_shapes[-2][1]
        curr_num_repeats = num_repeats[0]
        up_out_channel_list = [in_channels]  # used in DownBlocks

        for i in range(1, num_heads):
            curr_up_block = UpBlock(
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
            in_channels_next = prev_out_shapes[-(i + 2)][1]
            curr_num_repeats = num_repeats[i]

        # create num_heads-1 DownBlocks
        self.down_blocks = nn.ModuleList()
        channels_list_down_blocks = channels_list[(num_heads - 1) :]
        num_repeats_down_blocks = num_repeats[(num_heads - 1) :]

        in_channels = out_channels
        downsample_out_channels = channels_list_down_blocks[0]
        in_channels_next = up_out_channel_list[0]
        out_channels = channels_list_down_blocks[1]
        curr_num_repeats = num_repeats_down_blocks[0]

        for i in range(1, num_heads):
            curr_down_block = DownBlock(
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

    def forward(self, x):
        x0 = x[-1]
        up_block_outs = []
        for i, up_block in enumerate(self.up_blocks):
            conv_out, x0 = up_block(x0, x[(-i + 2)])
            up_block_outs.append(conv_out)
        up_block_outs.reverse()

        outs = [x0]
        for i, down_block in enumerate(self.down_blocks):
            x0 = down_block(x0, up_block_outs[i])
            outs.append(x0)
        return outs

    def _fit_to_num_heads(self, channels_list: list, num_repeats: list):
        """Fits channels_list and num_repeats to num_heads by removing or adding items"""
        if self.num_heads == 3:
            pass
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


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_channels_next: int,
        out_channels: int,
        num_repeats: int,
    ):
        """UpBlock used in RepPAN neck

        Args:
            in_channels (int): Number of input channels
            in_channels_next (int): Number of input channels of next input which is used in concat
            out_channels (int): Number of output channels
            num_repeats (int): Number of RepBlock repeats
        """
        super().__init__()

        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.rep_block = RepBlock(
            in_channels=in_channels_next + out_channels,
            out_channels=out_channels,
            n=num_repeats,
        )

    def forward(self, x0, x1):
        conv_out = self.conv(x0)
        upsample_out = self.upsample(conv_out)
        concat_out = torch.cat([upsample_out, x1], dim=1)
        out = self.rep_block(concat_out)
        return conv_out, out


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        in_channels_next: int,
        out_channels: int,
        num_repeats: int,
    ):
        """DownBlock used in RepPAN neck

        Args:
            in_channels (int): Number of input channels
            downsample_out_channels (int): Number of output channels after downsample
            in_channels_next (int): Number of input channels of next input which is used in concat
            out_channels (int): Number of output channels
            num_repeats (int): Number of RepBlock repeats
        """
        super().__init__()

        self.downsample = ConvModule(
            in_channels=in_channels,
            out_channels=downsample_out_channels,
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )
        self.rep_block = RepBlock(
            in_channels=downsample_out_channels + in_channels_next,
            out_channels=out_channels,
            n=num_repeats,
        )

    def forward(self, x0, x1):
        x = self.downsample(x0)
        x = torch.cat([x, x1], dim=1)
        x = self.rep_block(x)
        return x
