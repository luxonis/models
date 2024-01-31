#
# Adapted from: https://arxiv.org/pdf/2207.02696.pdf
#

import math
import warnings
from typing import List, Optional, cast

import torch
from torch import Tensor, nn

from luxonis_train.models.modules.luxonis_module import LuxonisModule
from luxonis_train.models.modules import (
    KeypointBlock,
    LearnableMulAddConv,
)
from luxonis_train.utils.boxutils import (
    process_bbox_predictions,
    process_keypoints_predictions,
)
from luxonis_train.utils.types import ModulePacket


class KeypointBboxHead(LuxonisModule):
    def validate(self, inputs: list[Tensor]) -> None:
        assert isinstance(inputs, list)
        assert len(inputs) == self.num_heads

    def __init__(
        self,
        n_classes: int,
        original_in_shape: list[int],
        n_keypoints: int = 17,
        num_heads: int = 3,
        anchors: List[List[int]] = [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401],
        ],
        visibility_threshold: float = 0.5,
        init_coco_biases: bool = True,
        **kwargs,
    ):
        """Head for for object and keypoint detection from
        `YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time
        object detectors`, https://arxiv.org/pdf/2207.02696.pdf

        Args:
            n_classes (int): Number of classes
            n_keypoints (int): Number of keypoints. Defaults to 17.
            num_heads (int): Number of output heads. Defaults to 3.
                ***Note:** Should be same also on neck in most cases.*
            anchors (List[List[int]]): Anchors used for object detection.
            Defaults to [ [12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401] ]. *(from COCO)*
            visibility_threshold (float, optional): Keypoints with visibility lower
            than threshold won't be drawn. Defaults to 0.5.
            init_coco_biases (bool, optional): Whether to use COCO bias and weight
            initialization. Defaults to True.
        """
        super().__init__(**kwargs)

        self.anchor_threshold = 4.0
        self.bias = 0.5
        self.original_in_shape = original_in_shape

        self.n_keypoints = n_keypoints
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.visibility_threshold = visibility_threshold

        self.box_offset = 5
        self.n_det_out = n_classes + self.box_offset
        self.n_kpt_out = 3 * n_keypoints
        self.n_out = self.n_det_out + self.n_kpt_out
        self.n_anchors = len(anchors[0]) // 2
        self.grid: List[Tensor] = []

        self.anchors = torch.tensor(anchors).float().view(self.num_heads, -1, 2)
        self.anchor_grid = self.anchors.clone().view(self.num_heads, 1, -1, 1, 1, 2)

        # TODO: remove this
        self.input_shapes = self.input_shapes[0]["features"]
        self.channel_list, self.stride = self._fit_to_num_heads(
            [c[1] for c in self.input_shapes]
        )

        self.learnable_mul_add_conv = nn.ModuleList(
            LearnableMulAddConv(
                add_channel=in_channels,
                mul_channel=self.n_det_out * self.n_anchors,
                conv_in_channel=in_channels,
                conv_out_channel=self.n_det_out * self.n_anchors,
            )
            for in_channels in self.channel_list
        )

        self.kpt_heads = nn.ModuleList(
            KeypointBlock(
                in_channels=in_channels,
                out_channels=self.n_kpt_out * self.n_anchors,
            )
            for in_channels in self.channel_list
        )

        self.anchors /= self.stride.view(-1, 1, 1)
        self._check_anchor_order()

        if init_coco_biases:
            self._initialize_weights_and_biases()

    def forward(self, inputs: list[ModulePacket]) -> ModulePacket:
        x = inputs[0]["features"]
        predictions: List[Tensor] = []
        features: List[Tensor] = []

        self.anchor_grid = self.anchor_grid.to(x[0].device)

        for i in range(self.num_heads):
            feat = cast(
                Tensor,
                torch.cat(
                    (
                        self.learnable_mul_add_conv[i](x[i]),
                        self.kpt_heads[i](x[i]),
                    ),
                    axis=1,
                ),  # type: ignore
            )

            batch_size, _, feature_height, feature_width = feat.shape
            if i >= len(self.grid):
                self.grid.append(
                    self._construct_grid(feature_width, feature_height).to(feat.device)
                )

            feat = feat.reshape(
                batch_size, self.n_anchors, self.n_out, feature_height, feature_width
            ).permute(0, 1, 3, 4, 2)

            features.append(feat)
            predictions.append(
                self._build_predictions(
                    feat, self.anchor_grid[i], self.grid[i], self.stride[i]
                )
            )

        # TODO: this can now be split to keypoints and boxes
        return {"keypoints": [torch.cat(predictions, 1)], "features": features}

    def _build_predictions(
        self, feat: Tensor, anchor_grid: Tensor, grid: Tensor, stride: Tensor
    ) -> Tensor:
        batch_size = feat.shape[0]
        x_bbox = feat[..., : self.box_offset + self.n_classes]
        x_keypoints = feat[..., self.box_offset + self.n_classes :]

        box_xy, box_wh, box_tail = process_bbox_predictions(x_bbox, anchor_grid)
        grid = grid.to(box_xy.device)
        stride = stride.to(box_xy.device)
        box_xy = (box_xy + grid) * stride
        out_bbox = torch.cat((box_xy, box_wh, box_tail), dim=-1)

        grid_x = grid[..., 0:1]
        grid_y = grid[..., 1:2]
        kpt_x, kpt_y, kpt_vis = process_keypoints_predictions(x_keypoints)
        kpt_x = (kpt_x + grid_x) * stride
        kpt_y = (kpt_y + grid_y) * stride
        out_kpt = torch.stack([kpt_x, kpt_y, kpt_vis.sigmoid()], dim=-1).reshape(
            *kpt_x.shape[:-1], -1
        )

        out = torch.cat((out_bbox, out_kpt), dim=-1)

        return out.reshape(batch_size, -1, self.n_out)

    def _infer_bbox(
        self, bbox: Tensor, stride: Tensor, grid: Tensor, anchor_grid: Tensor
    ) -> Tensor:
        out_bbox = bbox.sigmoid()
        out_bbox_xy = (out_bbox[..., 0:2] * 2.0 - 0.5 + grid) * stride
        out_bbox_wh = (out_bbox[..., 2:4] * 2) ** 2 * anchor_grid.view(
            1, self.n_anchors, 1, 1, 2
        )
        return torch.cat((out_bbox_xy, out_bbox_wh, out_bbox[..., 4:]), dim=-1)

    def _fit_to_num_heads(self, channel_list: list):
        out_channel_list = channel_list[: self.num_heads]
        # dynamically compute stride
        stride = torch.tensor(
            [
                self.original_in_shape[2] / x[2]
                for x in self.input_shapes[: self.num_heads]
            ],
            dtype=torch.int,
        )
        return out_channel_list, stride

    def _initialize_weights_and_biases(self, class_freq: Optional[torch.Tensor] = None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

        for mi, s in zip(self.learnable_mul_add_conv, self.stride):
            b = mi.conv.bias.view(self.n_anchors, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += (
                math.log(0.6 / (self.n_classes - 0.99))
                if class_freq is None
                else torch.log(class_freq / class_freq.sum())
            )
            mi.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _construct_grid(self, feature_width: int, feature_height: int):
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(feature_height), torch.arange(feature_width)], indexing="ij"
        )
        return (
            torch.stack((grid_x, grid_y), 2)
            .view((1, 1, feature_height, feature_width, 2))
            .float()
        )

    def _check_anchor_order(self):
        a = self.anchor_grid.prod(-1).view(-1)
        delta_a = a[-1] - a[0]
        delta_s = self.stride[-1] - self.stride[0]
        if delta_a.sign() != delta_s.sign():
            warnings.warn("Reversing anchor order")
            self.anchors[:] = self.anchors.flip(0)
            self.anchor_grid[:] = self.anchor_grid.flip(0)
