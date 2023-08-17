#
# Adapted from: https://github.com/WongKinYiu/yolov7
# License: https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md
#

import torch
import torch.nn as nn
import math
from typing import Optional, List
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes, draw_keypoints

from luxonis_ml.loader import LabelType
from luxonis_train.models.heads.base_heads import BaseHead
from luxonis_train.models.modules import ConvModule, autopad
from luxonis_train.utils.constants import HeadType
from luxonis_train.utils.boxutils import non_max_suppression_kpts


class IKeypointHead(BaseHead):
    head_types: List[HeadType] = [
        HeadType.OBJECT_DETECTION,
        HeadType.KEYPOINT_DETECTION,
    ]
    label_types: List[LabelType] = [LabelType.BOUNDINGBOX, LabelType.KEYPOINT]

    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list,
        n_keypoints: int,
        anchors: list,
        connectivity: Optional[list] = None,
        visibility_threshold: float = 0.5,
        attach_index: int = 0,
        main_metric: str = "map",
        **kwargs,
    ):
        """IKeypoint head which is used for object and keypoint detection

        Args:
            n_classes (int): Number of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original input shape to the model
            n_keypoints (int): Number of keypoints
            anchors (list): Anchors used for object detection
            connectivity (Optional[list], optional): Connectivity mapping used in visualization. Defaults to None.
            visibility_threshold (float, optional): Keypoints with visibility lower than threshold won't be drawn. Defaults to 0.5.
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

        self._validate_num_heads_and_attach_index(len(anchors))

        self.n_keypoints = n_keypoints
        self.connectivity = connectivity
        self.visibility_threshold = visibility_threshold

        self.n_det_out = self.n_classes + 5
        self.n_kpt_out = 3 * self.n_keypoints
        self.n_out = self.n_det_out + self.n_kpt_out
        self.num_heads = len(anchors)
        self.n_anchors = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.num_heads

        self.anchors = torch.tensor(anchors).float().view(self.num_heads, -1, 2)
        self.anchor_grid = self.anchors.clone().view(self.num_heads, 1, -1, 1, 1, 2)

        self.channel_list, self.stride = self._fit_to_num_heads(
            [c[1] for c in self.input_channels_shapes]
        )

        self.det_conv = nn.ModuleList(
            nn.Conv2d(in_channels, self.n_det_out * self.n_anchors, 1)
            for in_channels in self.channel_list
        )

        self.implicit_feat = nn.ModuleList(
            ImplicitFeatures(in_channels) for in_channels in self.channel_list
        )
        self.implicit_det = nn.ModuleList(
            ImplicitDetections(self.n_det_out * self.n_anchors)
            for _ in self.channel_list
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
        self._initialize_weights_and_biases()

    def forward(self, inputs):
        outs = []  # predictions
        x = []  # features

        if self.anchor_grid.device != inputs[self.attach_index].device:
            self.anchor_grid = self.anchor_grid.to(inputs[self.attach_index].device)

        for i in range(self.num_heads):
            x.append(
                torch.cat(
                    (
                        self.implicit_det[i](
                            self.det_conv[i](
                                self.implicit_feat[i](inputs[self.attach_index + i])
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

        return torch.cat(outs, 1), x

    def postprocess_for_loss(self, output: tuple, label_dict: dict):
        kpts = label_dict[LabelType.KEYPOINT]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        label = torch.zeros((len(boxes), nkpts * 2 + 6))
        label[:, :2] = boxes[:, :2]
        label[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "cxcywh")
        label[:, 6::2] = kpts[:, 2::3]  # insert kp x coordinates
        label[:, 7::2] = kpts[:, 3::3]  # insert kp y coordinates
        return output, label

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
                        self.implicit_det[i](
                            self.det_conv[i](
                                self.implicit_feat[i](inputs[self.attach_index + i])
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

    def _validate_num_heads_and_attach_index(self, num_heads: int):
        """Checks if specified number of heads is supported and if cumulative offset is valid"""
        if num_heads not in [2, 3, 4]:
            raise ValueError(
                "Specified number of heads not supported. Choose one of [2,3,4]"
            )

        if self.attach_index < 0:
            raise ValueError("Value of attach_index must be non-negative")

        if len(self.input_channels_shapes) - (self.attach_index + num_heads) < 0:
            raise ValueError("Cumulative offset (attach_index+num_head) out of range.")

    def _fit_to_num_heads(self, channel_list: list):
        """Returns correct channel list and stride based on num_heads and attach_index"""
        out_channel_list = channel_list[self.attach_index :][: self.num_heads]

        base_stride = [4, 8, 16, 32]
        stride = torch.tensor(base_stride[self.attach_index :][: self.num_heads])

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

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _check_anchor_order(self):
        a = self.anchor_grid.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = self.stride[-1] - self.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            print("Reversing anchor order")
            self.anchors[:] = self.anchors.flip(0)
            self.anchor_grid[:] = self.anchor_grid.flip(0)


class ImplicitFeatures(nn.Module):
    def __init__(self, channel: int):
        """Implicit block used for features"""
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self, x):
        return self.implicit.expand_as(x) + x


class ImplicitDetections(nn.Module):
    def __init__(self, channel: int):
        """Implicit block used for detection output"""
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x):
        return self.implicit.expand_as(x) * x


class KeypointBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Keypoint head block for keypoint predictions"""
        super().__init__()
        layers = []
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

    def forward(self, x):
        out = self.block(x)
        return out


if __name__ == "__main__":
    from luxonis_train.models.backbones import EfficientRep
    from luxonis_train.models.necks import RepPANNeck
    from luxonis_train.utils.general import dummy_input_run

    input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
    # input_shapes = [[1, 3, 256, 256]]
    backbone = EfficientRep()

    # num_heads_offset_pairs = [(2, [0, 1, 2]), (3, [0, 1]), (4, [0])]
    num_heads_offset_pairs = [(4, [-1])]

    for input_shape in input_shapes:
        for num_heads, offsets in num_heads_offset_pairs:
            for offset in offsets:
                input = torch.zeros(input_shape)
                input_channels_shapes = dummy_input_run(backbone, input_shape)
                backbone.eval()
                neck = RepPANNeck(
                    input_channels_shapes=input_channels_shapes,
                    num_heads=num_heads,
                    attach_index=offset,
                )
                input_channels_shapes = dummy_input_run(
                    neck, input_channels_shapes, multi_input=True
                )
                neck.eval()

                head = IKeypointHead(
                    n_classes=10,
                    input_channels_shapes=input_channels_shapes,
                    original_in_shape=input_shape,
                    n_keypoints=17,
                    anchors=[
                        [12, 16, 19, 36, 40, 28],
                        [36, 75, 76, 55, 72, 146],
                        # [142, 110, 192, 243, 459, 401],
                    ],
                    attach_index=2,
                )

                outs = backbone(input)
                outs = neck(outs)
                outs = head(outs)

                print(
                    "Keypoints:",
                    [o.shape for o in outs[0]],
                    "\nFeatures:",
                    [o.shape for o in outs[1]],
                )
