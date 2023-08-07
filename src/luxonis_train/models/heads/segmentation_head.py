#
# Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
# License: https://github.com/pytorch/vision/blob/main/LICENSE
#


import math
import warnings
import torch.nn as nn

from luxonis_train.models.modules import Up
from luxonis_train.models.heads.base_heads import BaseSegmentationHead


class SegmentationHead(BaseSegmentationHead):
    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        **kwargs
    ):
        """Basic segmentation FCN head. Note that it doesn't ensure that ouptut is same size as input.

        Args:
            n_classes (int): NUmber of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original inpuut shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
        """

        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        in_height = self.prev_out_shapes[self.attach_index][2]
        original_height = self.original_in_shape[2]
        num_up = math.log2(original_height) - math.log2(in_height)

        if not num_up.is_integer():
            warnings.warn(
                "Segmentation head's output shape not same as original input shape."
            )
            num_up = round(num_up)

        modules = []
        in_channels = self.prev_out_shapes[self.attach_index][1]
        for _ in range(int(num_up)):
            modules.append(Up(in_channels=in_channels, out_channels=in_channels // 2))
            in_channels //= 2

        self.head = nn.Sequential(
            *modules, nn.Conv2d(in_channels, n_classes, kernel_size=1)
        )

    def forward(self, x):
        out = self.head(x[self.attach_index])
        return out
