#
# Adapted from: https://github.com/WongKinYiu/yolov7
# License: https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md
#

import math
from typing import List

import torch
import torch.nn as nn

from luxonis_train.models.heads.ikeypoint_head import IKeypoint
from luxonis_train.utils.constants import HeadType


class YoloV7PoseHead(IKeypoint):
    head_types: List[HeadType] = [
        HeadType.OBJECT_DETECTION,
        HeadType.KEYPOINT_DETECTION,
    ]

    def __init__(
        self,
        anchors,
        **kwargs,
    ):
        super().__init__(
            anchors=anchors,
            **kwargs,
        )

        self.head = nn.ModuleList()

        for i in range(self.nl):
            curr_head = IKeypoint(
                prev_out_shapes=[self.prev_out_shapes[i]],
                n_classes=self.n_classes,
                n_keypoints=self.n_keypoints,
                original_in_shape=self.original_in_shape,
                anchors=[anchors[i]],
                main_metric=self.main_metric,
                connectivity=self.connectivity,
            )
            self.initialize_weights(curr_head)
            self.head.append(curr_head)

    def forward(self, x):
        z = []  # inference output
        out = []

        for i, module in enumerate(self.head):
            out_z, out_x = module([x[i]])
            out += out_x
            z.append(out_z)

        z = torch.cat(z, axis=1)  # type: ignore
        return z, out

    def initialize_weights(self, model, cf=None):
        for m in model.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

        # biases
        s = model.stride
        for mi in model.m:  # from
            b = mi.bias.view(model.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (model.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
