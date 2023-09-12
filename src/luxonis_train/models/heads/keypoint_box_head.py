import math
import warnings
from typing import Dict, List, Optional, Tuple, cast

import torch
from luxonis_ml.loader import LabelType
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes, draw_keypoints

from luxonis_train.models.heads.base_heads import BaseHead
from luxonis_train.models.modules import ConvModule, autopad
from luxonis_train.utils.boxutils import match_to_anchor, non_max_suppression_kpts
from luxonis_train.utils.constants import HeadType


class KeypointBoxHead(BaseHead):
    head_types: List[HeadType] = [
        HeadType.OBJECT_DETECTION,
        HeadType.KEYPOINT_DETECTION,
    ]
    label_types: List[LabelType] = [LabelType.BOUNDINGBOX, LabelType.KEYPOINT]

    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: List[List[int]],
        original_in_shape: List[int],
        n_keypoints: int,
        num_heads: int = 3,
        anchors: List[List[int]] = [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401],
        ],
        connectivity: Optional[List[int]] = None,
        visibility_threshold: float = 0.5,
        init_coco_biases: bool = True,
        attach_index: int = 0,
        main_metric: str = "map",
        **kwargs,
    ):
        """Head for for object and keypoint detection

        Args:
            n_classes (int): Number of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original input shape to the model
            n_keypoints (int): Number of keypoints
            num_heads (int): Number of output heads. Defaults to 3.
                ***Note:** Should be same also on neck in most cases.*
            anchors (list): Anchors used for object detection. Defaults to [ [12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401] ]. *(from COCO)*
            connectivity (Optional[list], optional): Connectivity mapping used in visualization. Defaults to None.
            visibility_threshold (float, optional): Keypoints with visibility lower than threshold won't be drawn. Defaults to 0.5.
            init_coco_biases (bool, optional): Weather to use COCO bias and weight initialization. Defaults to True.
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to 0.
                ***Note:** Value must be non-negative.**
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "map".
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
            main_metric=main_metric,
            **kwargs,
        )

        self._validate_params(num_heads, anchors)

        # TODO: customize
        self.anchor_threshold = 4.0
        self.bias = 0.5

        self.n_keypoints = n_keypoints
        self.num_heads = num_heads
        self.connectivity = connectivity
        self.visibility_threshold = visibility_threshold

        self.n_det_out = self.n_classes + 5
        self.n_kpt_out = 3 * self.n_keypoints
        self.n_out = self.n_det_out + self.n_kpt_out
        self.n_anchors = len(anchors[0]) // 2
        self.box_offset = 5
        self.grid: List[Tensor] = []

        self.anchors = torch.tensor(anchors).float().view(self.num_heads, -1, 2)
        self.anchor_grid = self.anchors.clone().view(self.num_heads, 1, -1, 1, 1, 2)

        self.channel_list, self.stride = self._fit_to_num_heads(
            [c[1] for c in self.input_channels_shapes]
        )

        self.det_conv = nn.ModuleList(
            nn.Conv2d(in_channels, self.n_det_out * self.n_anchors, 1)
            for in_channels in self.channel_list
        )

        self.implicit_add = nn.ModuleList(
            ImplicitAdd(in_channels) for in_channels in self.channel_list
        )
        self.implicit_mul = nn.ModuleList(
            ImplicitMultiply(self.n_det_out * self.n_anchors) for _ in self.channel_list
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

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        predictions: List[Tensor] = []
        features: List[Tensor] = []

        if self.anchor_grid.device != inputs[self.attach_index].device:
            self.anchor_grid = self.anchor_grid.to(inputs[self.attach_index].device)

        for i in range(self.num_heads):
            feat = cast(
                Tensor,
                torch.cat(
                    (
                        self.implicit_mul[i](
                            self.det_conv[i](
                                self.implicit_add[i](inputs[self.attach_index + i])
                            )
                        ),
                        self.kpt_heads[i](inputs[self.attach_index + i]),
                    ),
                    axis=1,
                ),  # type: ignore
            )

            batch_size, _, feature_height, feature_width = feat.shape
            if i >= len(self.grid):
                self.grid.append(
                    self._make_grid(feature_width, feature_height).to(feat.device)
                )

            feat = feat.reshape(
                batch_size, self.n_anchors, self.n_out, feature_height, feature_width
            ).permute(0, 1, 3, 4, 2)

            features.append(feat)

            x_bbox = feat[..., : self.box_offset + self.n_classes]
            x_keypoints = feat[..., self.box_offset + self.n_classes :]

            out_bbox = self._infer_bbox(
                x_bbox, self.stride[i], self.grid[i], self.anchor_grid[i]
            )

            out_kpt = self._infer_keypoints(x_keypoints, self.stride[i], self.grid[i])

            out = torch.cat((out_bbox, out_kpt), dim=-1)

            predictions.append(out.reshape(batch_size, -1, self.n_out))

        return torch.cat(predictions, 1), features

    def _infer_keypoints(
        self, keypoints: Tensor, stride: Tensor, grid: Tensor
    ) -> Tensor:
        grid_x = grid[..., 0:1]
        grid_y = grid[..., 1:2]

        x = (
            keypoints[..., ::3] * 2.0
            - 0.5
            + grid_x.repeat(1, 1, 1, 1, self.n_keypoints)
        ) * stride
        y = (
            keypoints[..., 1::3] * 2.0
            - 0.5
            + grid_y.repeat(1, 1, 1, 1, self.n_keypoints)
        ) * stride
        visibility = keypoints[..., 2::3].sigmoid()
        return torch.stack([x, y, visibility], dim=-1).reshape(*x.shape[:-1], -1)

    def _infer_bbox(
        self, bbox: Tensor, stride: Tensor, grid: Tensor, anchor_grid: Tensor
    ) -> Tensor:
        out_bbox = bbox.sigmoid()
        out_bbox_xy = (out_bbox[..., 0:2] * 2.0 - 0.5 + grid) * stride
        out_bbox_wh = (out_bbox[..., 2:4] * 2) ** 2 * anchor_grid.view(
            1, self.n_anchors, 1, 1, 2
        )
        return torch.cat((out_bbox_xy, out_bbox_wh, out_bbox[..., 4:]), dim=-1)

    def postprocess_for_loss(
        self, output: Tuple[Tensor, List[Tensor]], label_dict: Dict[str, Tensor]
    ) -> Tuple[
        Tuple[Tensor, List[Tensor]],
        Tuple[
            List[Tensor],
            List[Tensor],
            List[Tensor],
            List[Tuple[Tensor, Tensor, Tensor, Tensor]],
            List[Tensor],
        ],
    ]:
        predictions = output[1]
        kpts = label_dict[LabelType.KEYPOINT]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        targets = torch.zeros((len(boxes), nkpts * 2 + 6))
        targets[:, :2] = boxes[:, :2]
        targets[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "cxcywh")
        targets[:, 6::2] = kpts[:, 2::3]  # insert kp x coordinates
        targets[:, 7::2] = kpts[:, 3::3]  # insert kp y coordinates

        n_targets = len(targets)

        class_targets: List[Tensor] = []
        box_targets: List[Tensor] = []
        keypoint_targets: List[Tensor] = []
        indices: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
        anchors: List[Tensor] = []

        anchor_indices = (
            torch.arange(self.n_anchors, device=targets.device, dtype=torch.float32)
            .reshape(self.n_anchors, 1)
            .repeat(1, n_targets)
            .unsqueeze(-1)
        )
        targets = torch.cat((targets.repeat(self.n_anchors, 1, 1), anchor_indices), 2)

        XY_SHIFTS = (
            torch.tensor(
                [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device
            ).float()
            * self.bias
        )

        for i in range(self.num_heads):
            anchor = self.anchors[i]
            feature_width, feature_height = predictions[i].shape[2:4]

            scaled_targets, xy_shifts = match_to_anchor(
                targets,
                anchor,
                XY_SHIFTS,
                feature_width,
                feature_height,
                self.n_keypoints,
                self.anchor_threshold,
                self.bias,
                self.box_offset,
            )

            batch_index, cls = scaled_targets[:, :2].long().T
            box_xy = scaled_targets[:, 2:4]
            box_wh = scaled_targets[:, 4:6]
            box_xy_deltas = (box_xy - xy_shifts).long()
            feature_x_index = box_xy_deltas[:, 0].clamp_(0, feature_width - 1)
            feature_y_index = box_xy_deltas[:, 1].clamp_(0, feature_height - 1)

            anchor_indices = scaled_targets[:, -1].long()
            indices.append(
                (
                    batch_index,
                    anchor_indices,
                    feature_y_index,
                    feature_x_index,
                )
            )
            class_targets.append(cls)
            box_targets.append(torch.cat((box_xy - box_xy_deltas, box_wh), 1))
            anchors.append(anchor[anchor_indices])

            keypoint_targets.append(
                self._create_keypoint_target(scaled_targets, box_xy_deltas)
            )

        return output, (class_targets, box_targets, keypoint_targets, indices, anchors)

    def _create_keypoint_target(self, scaled_targets: Tensor, box_xy_deltas: Tensor):
        keypoint_target = scaled_targets[:, self.box_offset + 1 : -1]
        for j in range(self.n_keypoints):
            low = 2 * j
            high = 2 * (j + 1)
            keypoint_mask = keypoint_target[:, low:high] != 0
            keypoint_target[:, low:high][keypoint_mask] -= box_xy_deltas[keypoint_mask]
        return keypoint_target

    def postprocess_for_metric(self, output: tuple, label_dict: dict):
        kpts = label_dict[LabelType.KEYPOINT]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        label = torch.zeros((len(boxes), nkpts * 3 + 6))
        label[:, :2] = boxes[:, :2]
        label[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "cxcywh")
        label[:, 6::3] = kpts[:, 2::3]  # insert kp x coordinates
        label[:, 7::3] = kpts[:, 3::3]  # insert kp y coordinates
        label[:, 8::3] = kpts[:, 4::3]  # insert kp visibility

        nms = non_max_suppression_kpts(output[0])
        output_list_map = []
        output_list_oks = []
        output_list_kpt_map = []
        label_list_map = []
        label_list_oks = []
        label_list_kpt_map = []
        image_size = self.original_in_shape[2:]

        for i in range(len(nms)):
            output_list_map.append(
                {
                    "boxes": nms[i][:, :4],
                    "scores": nms[i][:, 4],
                    "labels": nms[i][:, 5].int(),
                }
            )
            output_list_oks.append({"keypoints": nms[i][:, 6:]})
            output_list_kpt_map.append(
                {
                    "boxes": nms[i][:, :4],
                    "scores": nms[i][:, 4],
                    "labels": nms[i][:, 5].int(),
                    "keypoints": nms[i][:, 6:],
                }
            )

            curr_label = label[label[:, 0] == i].to(nms[i].device)
            curr_bboxs = box_convert(curr_label[:, 2:6], "cxcywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list_map.append(
                {
                    "boxes": curr_bboxs,
                    "labels": curr_label[:, 1].int(),
                }
            )
            curr_kpts = curr_label[:, 6:]
            curr_kpts[:, 0::3] *= image_size[1]
            curr_kpts[:, 1::3] *= image_size[0]
            curr_bboxs_widths = curr_bboxs[:, 2] - curr_bboxs[:, 0]
            curr_bboxs_heights = curr_bboxs[:, 3] - curr_bboxs[:, 1]
            curr_scales = torch.sqrt(curr_bboxs_widths * curr_bboxs_heights)
            label_list_oks.append({"keypoints": curr_kpts, "scales": curr_scales})
            label_list_kpt_map.append(
                {
                    "boxes": curr_bboxs,
                    "labels": curr_label[:, 1].int(),
                    "keypoints": curr_kpts,
                }
            )

        # metric mapping is needed here because each metrics requires different output/label format
        metric_mapping = {"map": 0, "oks": 1, "kpt_map": 2}
        return (
            (output_list_map, output_list_oks, output_list_kpt_map),
            (label_list_map, label_list_oks, label_list_kpt_map),
            metric_mapping,
        )

    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        curr_output = output[0][idx]
        nms = non_max_suppression_kpts(
            curr_output.unsqueeze(0), conf_thresh=0.25, iou_thresh=0.45
        )[0]
        bboxes = nms[:, :4]
        img = draw_bounding_boxes(img, bboxes)
        kpts = nms[:, 6:].reshape(-1, self.n_keypoints, 3)
        # set coordinates of non-visible keypoints to (0, 0)
        mask = kpts[:, :, 2] < self.visibility_threshold
        kpts = kpts[:, :, 0:2] * (~mask).unsqueeze(-1).float()
        img = draw_keypoints(
            img, kpts[..., :2], colors="red", connectivity=self.connectivity
        )
        return img

    def get_output_names(self, idx: int):
        # TODO: check if this is correct output name
        return f"output{idx}"

    def forward_deploy(self, inputs):
        outs = []  # predictions
        x = []  # features

        if self.anchor_grid.device != inputs[self.attach_index].device:
            self.anchor_grid = self.anchor_grid.to(inputs[self.attach_index].device)

        for i in range(self.num_heads):
            x.append(
                torch.cat(
                    (
                        self.implicit_mul[i](
                            self.det_conv[i](
                                self.implicit_add[i](inputs[self.attach_index + i])
                            )
                        ),
                        self.kpt_heads[i](inputs[self.attach_index + i]),
                    ),
                    axis=1,
                )
            )

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.n_anchors, self.n_out, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            x_det = x[i][..., : 5 + self.n_classes]
            x_kpt = x[i][..., 5 + self.n_classes :]

            # from this point down only needed for inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            kpt_grid_x = self.grid[i][..., 0:1]
            kpt_grid_y = self.grid[i][..., 1:2]

            # det inference
            out_det = x_det.sigmoid()
            out_det_xy = (out_det[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
            out_det_wh = (out_det[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                1, self.n_anchors, 1, 1, 2
            )

            # kpt inference
            out_kpt_x = (
                x_kpt[..., ::3] * 2.0
                - 0.5
                + kpt_grid_x.repeat(1, 1, 1, 1, self.n_keypoints)
            ) * self.stride[i]
            out_kpt_y = (
                x_kpt[..., 1::3] * 2.0
                - 0.5
                + kpt_grid_y.repeat(1, 1, 1, 1, self.n_keypoints)
            ) * self.stride[i]
            out_kpt_cls = x_kpt[..., 2::3].sigmoid()
            out_kpt = torch.stack([out_kpt_x, out_kpt_y, out_kpt_cls], dim=-1).view(
                *out_kpt_x.shape[:-1], -1
            )

            out = torch.cat((out_det_xy, out_det_wh, out_det[..., 4:], out_kpt), dim=-1)
            outs.append(out.view(bs, -1, self.n_out))

        return torch.cat(outs, 1)

    def to_deploy(self):
        self.forward = self.forward_deploy

    def _validate_params(self, num_heads: int, anchors: List[List[int]]):
        """Checks num_heads, cumultive offset and anchors"""
        if num_heads not in [2, 3, 4]:
            raise ValueError(
                "Specified number of heads not supported. Choose one of [2,3,4]"
            )

        if len(anchors) != num_heads:
            raise ValueError(
                f"Number of anchors ({len(anchors)}) doesn't match number of heads ({num_heads})"
            )

        if self.attach_index < 0:
            raise ValueError("Value of attach_index must be non-negative")

        if len(self.input_channels_shapes) - (self.attach_index + num_heads) < 0:
            raise ValueError("Cumulative offset (attach_index+num_head) out of range.")

    def _fit_to_num_heads(self, channel_list: list):
        """Returns correct channel list and stride based on num_heads and attach_index"""
        out_channel_list = channel_list[self.attach_index :][: self.num_heads]
        # dynamically compute stride
        stride = torch.tensor(
            [
                self.original_in_shape[2] / x[2]
                for x in self.input_channels_shapes[self.attach_index :][
                    : self.num_heads
                ]
            ],
            dtype=int,
        )
        return out_channel_list, stride

    def _initialize_weights_and_biases(self, class_freq: Optional[torch.Tensor] = None):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

        # biases
        for mi, s in zip(self.det_conv, self.stride):  # from
            b = mi.bias.view(self.n_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (self.n_classes - 0.99))
                if class_freq is None
                else torch.log(class_freq / class_freq.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _make_grid(self, feature_width: int, feature_height: int):
        yv, xv = torch.meshgrid(
            [torch.arange(feature_height), torch.arange(feature_width)], indexing="ij"
        )
        return (
            torch.stack((xv, yv), 2)
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


class ImplicitAdd(nn.Module):
    def __init__(self, channel: int):
        """Implicit add block"""
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self, x: Tensor):
        return self.implicit.expand_as(x) + x


class ImplicitMultiply(nn.Module):
    def __init__(self, channel: int):
        """Implicit multiply block"""
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x: Tensor):
        return self.implicit.expand_as(x) * x


class KeypointBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Keypoint head block for keypoint predictions"""
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(6):
            depth_wise_conv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=autopad(3),
                groups=math.gcd(in_channels, in_channels),
                activation=nn.SiLU(),
            )
            conv = (
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    padding=autopad(1),
                    activation=nn.SiLU(),
                )
                if i < 5
                else nn.Conv2d(in_channels, out_channels, 1)
            )

            layers.append(depth_wise_conv)
            layers.append(conv)

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        out = self.block(x)
        return out
