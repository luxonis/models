#
# Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
# License: https://github.com/pytorch/vision/blob/main/LICENSE
#


import math
import warnings

import torch.nn as nn
from torch import Tensor

from luxonis_train.models.modules.luxonis_module import LuxonisModule
from luxonis_train.models.modules import UpBlock
from luxonis_train.utils.types import ModulePacket


class SegmentationHead(LuxonisModule):
    def __init__(
        self,
        n_classes: int,
        original_in_shape: list[int],
        attach_index: int = -1,
        **kwargs
    ):
        """Basic segmentation FCN head. Note that it doesn't ensure
        that ouptut is same size as input.

        Args:
            n_classes (int): NUmber of classes
            attach_index (int, optional): Index of previous output that the
            head attaches to. Defaults to -1.
        """
        super().__init__(**kwargs)

        self.original_in_shape = original_in_shape
        self.attach_index = attach_index
        # TODO: change this
        self.input_shapes = self.input_shapes[0]["features"]
        in_height = self.input_shapes[self.attach_index][2]
        original_height = self.original_in_shape[2]
        num_up = math.log2(original_height) - math.log2(in_height)

        if not num_up.is_integer():
            warnings.warn(
                "Segmentation head's output shape not same as original input shape."
            )
            num_up = round(num_up)

        modules = []
        in_channels = self.input_shapes[self.attach_index][1]
        for _ in range(int(num_up)):
            modules.append(
                UpBlock(in_channels=in_channels, out_channels=in_channels // 2)
            )
            in_channels //= 2

        self.head = nn.Sequential(
            *modules, nn.Conv2d(in_channels, n_classes, kernel_size=1)
        )

    def validate(self, inputs: list[Tensor]) -> None:
        assert len(inputs) > self.attach_index or self.attach_index < 0

    def forward(self, x: list[ModulePacket]) -> ModulePacket:
        out = self.head(x[0]["features"][self.attach_index])
        return {"segmentation": [out]}
