#
# Adapted from: https://github.com/WongKinYiu/yolov7
# License: https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md
#

import math
import torch
import torch.nn as nn

from luxonis_train.models.heads.ikeypoint_head import IKeypointHead


class IKeypointMultiHead(IKeypointHead):
    def __init__(
        self,
        anchors: list,
        **kwargs,
    ):
        """Object and keypoint detection head with separate IKeypoint head for each layer of anchors.

        Args:
            anchors (list): Anchors used for object detection
            n_classes (int): Number of classes
            n_keypoints (int): Number of keypoints
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "map".
            connectivity (Optional[list], optional): Connectivity mapping used in visualization. Defaults to None.
            visibility_threshold (float, optional): Keypoints with visibility lower than threshold won't be drawn. Defaults to 0.5.
        """
        super().__init__(
            anchors=anchors,
            **kwargs,
        )

        self.heads = nn.ModuleList()

        for i in range(self.num_heads):
            curr_head = IKeypointHead(
                input_channels_shapes=[self.input_channels_shapes[i]],
                n_classes=self.n_classes,
                n_keypoints=self.n_keypoints,
                original_in_shape=self.original_in_shape,
                anchors=[anchors[i]],
                main_metric=self.main_metric,
                connectivity=self.connectivity,
                visibility_threshold=self.visibility_threshold,
            )
            self.initialize_weights(curr_head)
            self.heads.append(curr_head)

    def forward(self, x):
        z = []  # inference output
        out = []

        for i, module in enumerate(self.heads):
            out_z, out_x = module([x[i]])
            out += out_x
            z.append(out_z)

        z = torch.cat(z, axis=1)  # type: ignore
        return z, out

    def initialize_weights(self, model, cf=None):
        for m in model.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

        # biases
        s = model.stride
        for mi in model.m:  # from
            b = mi.bias.view(model.n_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (model.n_classes - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
