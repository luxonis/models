#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#

import torch
import torch.nn as nn
import math

from luxonis_train.models.modules import ConvModule
from luxonis_train.models.heads.base_heads import BaseObjectDetection


class EffiDeHead(BaseObjectDetection):
    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        n_anchors: int = 1,
    ):
        """EffieDeHead object detection head which is part of YoloV6 head

        Args:
            n_classes (int): Number of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            n_anchors (int, optional): Should stay default. Defaults to 1.
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.n_anchors = n_anchors
        self.prior_prob = 1e-2

        in_channels = self.input_channels_shapes[self.attach_index][1]
        self.head = nn.Sequential(
            *[
                # stem
                ConvModule(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    activation=nn.SiLU(),
                ),
                # cls_conv
                ConvModule(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    activation=nn.SiLU(),
                ),
                # reg_conv
                ConvModule(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    activation=nn.SiLU(),
                ),
                # cls_pred
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.n_classes * self.n_anchors,
                    kernel_size=1,
                ),
                # reg_pred
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=4 * (self.n_anchors),
                    kernel_size=1,
                ),
            ]
        )
        self.initialize_biases()

    def initialize_biases(self):
        # cls_pred
        conv = self.head[3]
        b = conv.bias.view(
            -1,
        )
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
        w = conv.weight
        w.data.fill_(0.0)
        conv.weight = nn.Parameter(w, requires_grad=True)

        # reg_pred
        conv = self.head[4]
        b = conv.bias.view(
            -1,
        )
        b.data.fill_(1.0)
        conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
        w = conv.weight
        w.data.fill_(0.0)
        conv.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        out = self.head[0](x[self.attach_index])
        out_cls = self.head[1](out)
        out_cls = self.head[3](out_cls)
        out_reg = self.head[2](out)
        out_reg = self.head[4](out_reg)

        return [x[-1], out_cls, out_reg]


if __name__ == "__main__":
    from luxonis_train.models.backbones import *
    from luxonis_train.utils.general import dummy_input_run

    backbone = ResNet18()
    backbone_out_shapes = dummy_input_run(backbone, [1, 3, 224, 224])
    backbone.eval()

    shapes = [224, 256, 384, 512]
    shapes = [512]
    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        head = EffiDeHead(
            prev_out_shape=backbone_out_shapes, n_classes=10, original_in_shape=x.shape
        )
        head.eval()
        outs = head(outs)
        for i in range(len(outs)):
            print(f"Output {i}:")
            if isinstance(outs[i], list):
                for o in outs[i]:
                    print(len(o) if isinstance(o, list) else o.shape)
            else:
                print(outs[i].shape)
