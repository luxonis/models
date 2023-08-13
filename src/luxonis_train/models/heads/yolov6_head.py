#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#

import torch
import torch.nn as nn
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from .effide_head import EffiDeHead

# from luxonis_train.models.heads.effide_head import EffiDeHead #import for unit testing
from luxonis_train.models.heads.base_heads import BaseObjectDetection
from luxonis_train.utils.assigners.anchor_generator import generate_anchors
from luxonis_train.utils.boxutils import dist2bbox, non_max_suppression_bbox


class YoloV6Head(BaseObjectDetection):
    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        is_4head: bool = False,
        **kwargs,
    ):
        """YoloV6 object detection head. With hardware-aware degisn, the decoupled head is optimized with
            hybridchannels methods.

        Args:
            n_classes (int): Number of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            is_4head (bool, optional): Either build 4 headed architecture or 3 headed one
                (**Important: Should be same also on backbone and neck**). Defaults to False.
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.no = n_classes + 5  # number of outputs per anchor
        self.is_4head = is_4head
        self.nl = (
            4 if self.is_4head else 3
        )  # number of detection layers (support 3 and 4 heads)

        self.prior_prob = 1e-2

        self.n_anchors = 1
        stride = (
            [4, 8, 16, 32] if self.is_4head else [8, 16, 32]
        )  # strides computed during build
        self.stride = torch.tensor(stride)
        self.grid = [torch.zeros(1)] * self.nl
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        self.head = nn.ModuleList()
        for i in range(self.nl):
            curr_head = EffiDeHead(
                input_channels_shapes=[self.input_channels_shapes[i]],
                original_in_shape=self.original_in_shape,
                n_classes=self.n_classes,
                n_anchors=self.n_anchors,
            )
            self.head.append(curr_head)

    def forward(self, x):
        cls_score_list = []
        reg_distri_list = []

        for i, module in enumerate(self.head):
            out_x, out_cls, out_reg = module([x[i]])
            x[i] = out_x
            out_cls = torch.sigmoid(out_cls)
            cls_score_list.append(out_cls.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(out_reg.flatten(2).permute((0, 2, 1)))

        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)

        return [x, cls_score_list, reg_distri_list]

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

    def draw_output_to_img(self, img: torch.Tensor, output: tuple, idx: int):
        curr_output = self._out2box(output, conf_thres=0.3, iou_thres=0.6)
        curr_output = curr_output[idx]
        bboxs = curr_output[:, :4]
        img = draw_bounding_boxes(img, bboxs)
        return img

    def get_output_names(self, idx: int):
        output_names = ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"]
        if self.is_4head:
            output_names.append("output4_yolov6r2")
        return output_names

    def to_deploy(self):
        # change definition of forward()
        def deploy_forward(x):
            outputs = []
            for i, module in enumerate(self.head):
                out_x, out_cls, out_reg = module([x[i]])
                out_cls = torch.sigmoid(out_cls)
                conf, _ = out_cls.max(1, keepdim=True)
                output = torch.cat([out_reg, conf, out_cls], axis=1)
                outputs.append(output)
            return outputs

        self.forward = deploy_forward

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


if __name__ == "__main__":
    # test yolov6-n config
    from luxonis_train.models.backbones import EfficientRep
    from luxonis_train.models.necks import RepPANNeck
    from luxonis_train.utils.general import dummy_input_run

    num_repeats_backbone = [1, 6, 12, 18, 6]
    num_repeats_neck = [12, 12, 12, 12]
    depth_mul = 0.33

    channels_list_backbone = [64, 128, 256, 512, 1024]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    width_mul = 0.25

    backbone = EfficientRep(
        in_channels=3,
        channels_list=channels_list_backbone,
        num_repeats=num_repeats_backbone,
        depth_mul=depth_mul,
        width_mul=width_mul,
        is_4head=True,
    )
    for module in backbone.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    backbone_out_shapes = dummy_input_run(backbone, [1, 3, 224, 224])
    backbone.eval()

    neck = RepPANNeck(
        prev_out_shape=backbone_out_shapes,
        channels_list=channels_list_neck,
        num_repeats=num_repeats_neck,
        depth_mul=depth_mul,
        width_mul=width_mul,
        is_4head=True,
    )
    neck_out_shapes = dummy_input_run(neck, backbone_out_shapes, multi_input=True)
    neck.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        head = YoloV6Head(
            prev_out_shape=neck_out_shapes,
            n_classes=10,
            original_in_shape=x.shape,
            is_4head=True,
        )
        head.eval()
        outs = backbone(x)
        outs = neck(outs)
        outs = head(outs)
        for i in range(len(outs)):
            print(f"Output {i}:")
            if isinstance(outs[i], list):
                for o in outs[i]:
                    print(o.shape)
            else:
                print(outs[i].shape)
