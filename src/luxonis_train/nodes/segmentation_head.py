"""Implementation of a basic segmentation head.

Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
License (BSD-3): https://github.com/pytorch/vision/blob/main/LICENSE
"""


from typing import cast

import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.blocks import UpBlock
from luxonis_train.utils.general import infer_upscale_factor
from luxonis_train.utils.types import Packet

from .base_node import BaseNode


class SegmentationHead(BaseNode[Tensor, Tensor]):
    def __init__(self, **kwargs):
        """Basic segmentation FCN head.

        Note that it doesn't ensure that ouptut is same size as input.
        """
        super().__init__(attach_index=-1, **kwargs)

        original_height = self.original_in_shape[2]
        num_up = infer_upscale_factor(
            cast(int, self.in_height), original_height, strict=False
        )

        modules = []
        in_channels = cast(int, self.in_channels)
        for _ in range(int(num_up)):
            modules.append(
                UpBlock(in_channels=in_channels, out_channels=in_channels // 2)
            )
            in_channels //= 2

        self.head = nn.Sequential(
            *modules,
            nn.Conv2d(in_channels, self.dataset_metadata.n_classes, kernel_size=1),
        )

    def wrap(self, output: Tensor) -> Packet[Tensor]:
        return {"segmentation": [output]}

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)