#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#

import math
import torch
import torch.nn as nn
from typing import Literal
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from luxonis_train.models.heads.base_heads import BaseObjectDetection
from luxonis_train.models.modules import ConvModule
from luxonis_train.utils.assigners.anchor_generator import generate_anchors
from luxonis_train.utils.boxutils import dist2bbox, non_max_suppression_bbox


class YoloV6Head(BaseObjectDetection):
    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list,
        num_heads: Literal[2, 3, 4] = 3,
        attach_index: int = 0,
        **kwargs,
    ):
        """YoloV6 object detection head. With hardware-aware degisn, the decoupled head is optimized with
            hybridchannels methods.

        Args:
            n_classes (int): Number of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original input shape to the model
            num_heads (Literal[2,3,4], optional): Number of output heads. Defaults to 3.
                ***Note:** Should be same also on neck in most cases.*
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to 0.
                ***Note:** Value must be non-negative.**
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
            **kwargs,
        )

        self._validate_num_heads_and_attach_index(num_heads)
        self.num_heads = num_heads

        self.stride = self._fit_stride_to_num_heads()
        self.grid = [torch.zeros(1)] * self.num_heads
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            curr_head = EfficientDecoupledBlock(
                n_classes=self.n_classes,
                in_channels=self.input_channels_shapes[self.attach_index + i][1],
            )
            self.heads.append(curr_head)

    def forward(self, x):
        feature_list = []
        cls_score_list = []
        reg_distri_list = []

        for i, module in enumerate(self.heads):
            out_x, out_cls, out_reg = module(x[self.attach_index + i])
            feature_list.append(out_x)
            out_cls = torch.sigmoid(out_cls)
            cls_score_list.append(out_cls.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(out_reg.flatten(2).permute((0, 2, 1)))

        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)

        return [feature_list, cls_score_list, reg_distri_list]

    def postprocess_for_loss(self, output: tuple, label_dict: dict):
        label = label_dict[self.label_types[0]]
        return output, label

    def postprocess_for_metric(self, output: tuple, label_dict: dict):
        label = label_dict[self.label_types[0]]

        output_nms = self._out2box(output)
        image_size = self.original_in_shape[2:]

        output_list = []
        label_list = []
        for i in range(len(output_nms)):
            output_list.append(
                {
                    "boxes": output_nms[i][:, :4],
                    "scores": output_nms[i][:, 4],
                    "labels": output_nms[i][:, 5].int(),
                }
            )

            curr_label = label[label[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list.append({"boxes": curr_bboxs, "labels": curr_label[:, 1].int()})

        return output_list, label_list, None

    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        curr_output = self._out2box(output, conf_thres=0.3, iou_thres=0.6)
        curr_output = curr_output[idx]
        bboxs = curr_output[:, :4]
        img = draw_bounding_boxes(img, bboxs)
        return img

    def get_output_names(self, idx: int):
        output_names = [f"output{i}_yolov6r2" for i in range(1, self.num_heads + 1)]
        return output_names

    def forward_deploy(self, x):
        outputs = []
        for i, module in enumerate(self.heads):
            _, out_cls, out_reg = module(x[self.attach_index + i])
            out_cls = torch.sigmoid(out_cls)
            conf, _ = out_cls.max(1, keepdim=True)
            output = torch.cat([out_reg, conf, out_cls], axis=1)
            outputs.append(output)
        return outputs

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

    def _fit_stride_to_num_heads(self):
        """Returns correct stride for number of heads and attach index"""
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
        return stride

    def _out2box(self, output: tuple, **kwargs):
        """Performs post-processing of the YoloV6 output and returns bboxs after NMS"""
        x, cls_score_list, reg_dist_list = output
        anchor_points, stride_tensor = generate_anchors(
            x, self.stride, self.grid_cell_size, self.grid_cell_offset, is_eval=True
        )
        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")

        pred_bboxes *= stride_tensor
        output_merged = torch.cat(
            [
                pred_bboxes,
                torch.ones(
                    (x[-1].shape[0], pred_bboxes.shape[1], 1),
                    dtype=pred_bboxes.dtype,
                    device=pred_bboxes.device,
                ),
                cls_score_list,
            ],
            axis=-1,
        )

        conf_thres = kwargs.get("conf_thres", 0.001)
        iou_thres = kwargs.get("iou_thres", 0.6)
        output_nms = non_max_suppression_bbox(
            output_merged, conf_thres=conf_thres, iou_thres=iou_thres
        )

        return output_nms


class EfficientDecoupledBlock(nn.Module):
    def __init__(self, n_classes: int, in_channels: int):
        """Efficient Decoupled block used for class and regression predictions.

        Args:
            n_classes (int): Number of classes
            in_channels (int): Number of input channels
        """
        super().__init__()

        self.stem = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            activation=nn.SiLU(),
        )

        self.cls_conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.SiLU(),
        )
        self.cls_pred = nn.Conv2d(
            in_channels=in_channels, out_channels=n_classes, kernel_size=1
        )

        self.reg_conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.SiLU(),
        )
        self.reg_pred = nn.Conv2d(
            in_channels=in_channels, out_channels=4, kernel_size=1
        )

        prior_prob = 1e-2
        self._initialize_weights_and_biases(prior_prob)

    def forward(self, x):
        out = self.stem(x)
        # class branch
        out_cls = self.cls_conv(out)
        out_cls = self.cls_pred(out_cls)
        # regression branch
        out_reg = self.reg_conv(out)
        out_reg = self.reg_pred(out_reg)

        return [out, out_cls, out_reg]

    def _initialize_weights_and_biases(self, prior_prob: float):
        data = [
            (self.cls_pred, -math.log((1 - prior_prob) / prior_prob)),
            (self.reg_pred, 1.0),
        ]
        for module, fill_value in data:
            b = module.bias.view(-1)
            b.data.fill_(fill_value)
            module.bias = nn.Parameter(b.view(-1), requires_grad=True)

            w = module.weight
            w.data.fill_(0.0)
            module.weight = nn.Parameter(w, requires_grad=True)
