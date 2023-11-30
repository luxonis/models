"""Head for object and keypoint detection.

Adapted from `YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time
object detectors`, available at `
https://arxiv.org/pdf/2207.02696.pdf`.
"""

import logging
import math
from typing import Literal

import torch
from torch import Tensor, nn
from typeguard import check_type

from luxonis_train.nodes.blocks import (
    KeypointBlock,
    LearnableMulAddConv,
)
from luxonis_train.utils.boxutils import (
    non_max_suppression,
    process_bbox_predictions,
    process_keypoints_predictions,
)
from luxonis_train.utils.types import Packet

from .base_node import BaseNode


class ImplicitKeypointBBoxHead(BaseNode):
    """Head for object and keypoint detection.

    TODO: more technical documentation
    """

    attach_index: Literal["all"] = "all"

    def __init__(
        self,
        n_classes: int | None = None,
        n_keypoints: int | None = None,
        num_heads: int = 3,
        anchors: list[list[int]] | None = None,
        init_coco_biases: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        **kwargs,
    ):
        """Constructor for the `ImplicitKeypointBBoxHead` module.

        Args:
            n_classes (int): Number of classes.
            n_keypoints (int): Number of keypoints.
            num_heads (int, optional): Number of output heads. Defaults to 3.
                ***Note:*** Should be same also on neck in most cases.*
            anchors (list[list[int]]): Anchors used for object detection.
              Defaults to:
              ```
              [ [12, 16, 19, 36, 40, 28],
                [36, 75, 76, 55, 72, 146],
                [142, 110, 192, 243, 459, 401],
              ]
              ```
              Inferred from the COCO dataset.
            than threshold won't be drawn. Defaults to 0.5.
            init_coco_biases (bool, optional): Whether to use COCO bias and weight
            initialization. Defaults to True.
        """
        super().__init__(**kwargs)
        anchors = anchors or [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401],
        ]

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.n_keypoints = n_keypoints or self.dataset_metadata.n_keypoints
        self.n_classes = n_classes or self.dataset_metadata.n_classes

        if self.n_keypoints == 0:
            raise ValueError(
                "Number of keypoints must be specified either in the constructor or "
                "in the dataset metadata."
            )
        if self.n_classes == 0:
            raise ValueError(
                "Number of classes must be specified either in the constructor or "
                "in the dataset metadata."
            )
        self.num_heads = num_heads

        self.box_offset = 5
        self.n_det_out = self.n_classes + self.box_offset
        self.n_kpt_out = 3 * self.n_keypoints
        self.n_out = self.n_det_out + self.n_kpt_out
        self.n_anchors = len(anchors[0]) // 2
        self.grid: list[Tensor] = []

        self.anchors = torch.tensor(anchors).float().view(self.num_heads, -1, 2)
        self.anchor_grid = self.anchors.clone().view(self.num_heads, 1, -1, 1, 1, 2)

        self.channel_list, self.stride = self._fit_to_num_heads(
            check_type(self.in_channels, list[int])
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

    def forward(self, inputs: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        predictions: list[Tensor] = []
        features: list[Tensor] = []

        self.anchor_grid = self.anchor_grid.to(inputs[0].device)

        for i in range(self.num_heads):
            feat = check_type(
                torch.cat(
                    (
                        self.learnable_mul_add_conv[i](inputs[i]),
                        self.kpt_heads[i](inputs[i]),
                    ),
                    axis=1,
                ),  # type: ignore
                Tensor,
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

        return features, torch.cat(predictions, dim=1)

    def wrap(self, outputs: tuple[list[Tensor], Tensor]) -> Packet[Tensor]:
        features, predictions = outputs

        if self.export:
            return {"boxes_and_keypoints": [predictions]}

        if self.training:
            return {"features": features}

        nms = non_max_suppression(
            predictions,
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="cxcywh",
        )

        return {
            "boxes": [detection[:, :6] for detection in nms],
            "keypoints": [
                detection[:, 6:].reshape(-1, self.n_keypoints, 3) for detection in nms
            ],
            "features": features,
        }

    def _build_predictions(
        self, feat: Tensor, anchor_grid: Tensor, grid: Tensor, stride: Tensor
    ) -> Tensor:
        batch_size = feat.shape[0]
        x_bbox = feat[..., : self.box_offset + self.n_classes]
        x_keypoints = feat[..., self.box_offset + self.n_classes :]

        box_cxcy, box_wh, box_tail = process_bbox_predictions(x_bbox, anchor_grid)
        grid = grid.to(box_cxcy.device)
        stride = stride.to(box_cxcy.device)
        box_cxcy = (box_cxcy + grid) * stride
        out_bbox = torch.cat((box_cxcy, box_wh, box_tail), dim=-1)

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
        stride = torch.tensor(
            [
                self.original_in_shape[2] / h
                for h in check_type(self.in_height, list[int])[: self.num_heads]
            ],
            dtype=torch.int,
        )
        return out_channel_list, stride

    def _initialize_weights_and_biases(self, class_freq: Tensor | None = None):
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
            logging.getLogger(__name__).warning("Reversing anchor order")
            self.anchors[:] = self.anchors.flip(0)
            self.anchor_grid[:] = self.anchor_grid.flip(0)
