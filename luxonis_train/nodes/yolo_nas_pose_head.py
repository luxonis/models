"""Yolo NAS Pose head.

YoloNASPoseNDFLHeads
Source: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/pose_estimation_models/yolo_nas_pose/yolo_nas_pose_ndfl_heads.py
License: Apache-2.0 license https://github.com/Deci-AI/super-gradients?tab=Apache-2.0-1-ov-file#readme
Note: Only the super-gradients source code is Apache-2.0, all Yolo NAS & Yolo NAS Pose weights are under a restrictive license
Weights license: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md
"""


import torchvision
import torch
from torch import Tensor
import torch.nn as nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks.yolo_nas_blocks import (
    YoloNASPoseDFLHead
)
from luxonis_train.utils.nas_utils import (
    generate_anchors_for_grid_cell,
    batch_distance2bbox,
    YoloNasPoseDecodedPredictions,
    YoloNasPoseRawOutputs
)

from typing import List, Optional, Tuple, Union


class YoloNASPoseHead(BaseNode[list[Tensor], list[Tensor]]):
    attach_index: int = -1

    VARIANTS_CONFIGS: dict[str, dict] = {
        "n": None,
        "s": {
            "config": {
                "reg_max": 16,
                "pose_offset_multiplier": 1.0,
                "compensate_grid_cell_offset": True,
                "inference_mode": False
            },
            "modules": {
                "head_0": {
                    "bbox_inter_channels": 128,
                    "pose_inter_channels": 128,
                    "pose_regression_blocks": 2,
                    "shared_stem": False,
                    "width_mult": 0.5,
                    "pose_conf_in_class_head": True,
                    "pose_block_use_repvgg": False,
                    "first_conv_group_size": 0,
                    "stride": 8,
                    "reg_max": 16,
                },
                "head_1": {
                    "bbox_inter_channels": 256,
                    "pose_inter_channels": 512,
                    "pose_regression_blocks": 2,
                    "shared_stem": False,
                    "width_mult": 0.5,
                    "pose_conf_in_class_head": True,
                    "pose_block_use_repvgg": False,
                    "first_conv_group_size": 0,
                    "stride": 16,
                    "reg_max": 16
                },
                "head_2": {
                    "bbox_inter_channels": 512,
                    "pose_inter_channels": 512,
                    "pose_regression_blocks": 3,
                    "shared_stem": False,
                    "width_mult": 0.5,
                    "pose_conf_in_class_head": True,
                    "pose_block_use_repvgg": False,
                    "first_conv_group_size": 0,
                    "stride": 32,
                    "reg_max": 16
                }
            },
            "in_channels": [96, 192, 384]
        },
        "m": None,
        "l": None,
    }

    def __init__(
        self,
        num_classes: int,
        variant: str,
        
        # handled by variant config
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        inference_mode: bool = False,
        eval_size: Optional[Tuple[int, int]] = None,
        width_mult: float = 1.0,
        pose_offset_multiplier: float = 1.0,
        compensate_grid_cell_offset: bool = True,
        **kwargs,
    ):
        """Simple wrapper for YoloNASPoseNDFLHeads head, source above ^^.

        Args:
            in_channels List(int): list of input feature maps channels.
            variant (str): Yolo NAS variant ["n", "s", "m", "l"].
        """
        super().__init__(**kwargs)
        
        if not variant in YoloNASPoseHead.VARIANTS_CONFIGS:
            raise ValueError(
                f"YoloNASPoseHead variant should be in {YoloNASPoseHead.VARIANTS_CONFIGS.keys()}"
            )

        self.variant_config = YoloNASPoseHead.VARIANTS_CONFIGS[variant]

        in_channels = self.variant_config["in_channels"]
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size
        self.pose_offset_multiplier = pose_offset_multiplier
        self.compensate_grid_cell_offset = compensate_grid_cell_offset
        self.inference_mode = inference_mode

        # update variant specific values
        for attr_name, attr_value in self.variant_config["config"].items():
            setattr(self, attr_name, attr_value)

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        self._init_weights()

        self.num_heads = len(self.variant_config["modules"])
        fpn_strides: List[int] = []
        for i in range(self.num_heads):
            head_config = self.variant_config["modules"][f"head_{i}"]
            new_head = YoloNASPoseDFLHead(in_channels=in_channels[i], num_classes=num_classes, **head_config)
            fpn_strides.append(new_head.stride)
            setattr(self, f"head{i + 1}", new_head)

        self.fpn_strides = tuple(fpn_strides)

    @torch.jit.ignore
    def _init_weights(self):
        if self.eval_size:

            device, dtype = None, None

            try:
                device = next(iter(self.parameters())).device
            except StopIteration:
            
                device =  next(iter(self.buffers())).device

            try:
                dtype = next(iter(self.parameters())).dtype
            except:
                dtype = next(iter(self.buffers())).dtype

            anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats: Tuple[Tensor, ...]) -> Union[YoloNasPoseDecodedPredictions, Tuple[YoloNasPoseDecodedPredictions, YoloNasPoseRawOutputs]]:
        """
        Runs the forward for all the underlying heads and concatenate the predictions to a single result.
        :param feats: List of feature maps from the neck of different strides
        :return: Return value depends on the mode:
        If tracing, a tuple of 4 tensors (decoded predictions) is returned:
        - pred_bboxes [B, Num Anchors, 4] - Predicted boxes in XYXY format
        - pred_scores [B, Num Anchors, 1] - Predicted scores for each box
        - pred_pose_coords [B, Num Anchors, Num Keypoints, 2] - Predicted poses in XY format
        - pred_pose_scores [B, Num Anchors, Num Keypoints] - Predicted scores for each keypoint

        In training/eval mode, a tuple of 2 tensors returned:
        - decoded predictions - they are the same as in tracing mode
        - raw outputs - a tuple of 8 elements in total, this is needed for training the model.
        """

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        pose_regression_list = []
        pose_logits_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit, pose_regression, pose_logits = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            pose_regression_list.append(torch.permute(pose_regression.flatten(3), [0, 3, 1, 2]))  # [B, J, 2, H, W] -> [B, H * W, J, 2]
            pose_logits_list.append(torch.permute(pose_logits.flatten(2), [0, 2, 1]))  # [B, J, H, W] -> [B, H * W, J]

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, J, 2]
        pose_logits_list = torch.cat(pose_logits_list, dim=1)  # [B, Anchors, J]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        # Decode keypoints
        if self.pose_offset_multiplier != 1.0:
            pose_regression_list *= self.pose_offset_multiplier

        if self.compensate_grid_cell_offset:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2) - self.grid_cell_offset
        else:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2)

        pose_regression_list *= stride_tensor.unsqueeze(0).unsqueeze(2)

        pred_pose_coords = pose_regression_list.detach().clone()  # [B, Anchors, C, 2]
        pred_pose_scores = pose_logits_list.detach().clone().sigmoid()  # [B, Anchors, C]

        decoded_predictions = pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores

        if torch.jit.is_tracing() or self.inference_mode:
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = cls_score_list, reg_distri_list, pose_regression_list, pose_logits_list, anchors, anchor_points, num_anchors_list, stride_tensor
        return decoded_predictions, raw_predictions

    @property
    def out_channels(self):
        return None

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []

        dtype = dtype or feats[0].dtype
        device = device or feats[0].device

        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)

        if device is not None:
            anchor_points = anchor_points.to(device)
            stride_tensor = stride_tensor.to(device)
        return anchor_points, stride_tensor


if __name__ == "__main__":

    from luxonis_train.nodes.yolo_nas_backbone import YoloNASBackbone
    from luxonis_train.nodes.yolo_nas_neck import YoloNASNeck

    # variant consistency between backbone/neck/head will be handled by using a single variant value from global config
    # only "s" variant is configured currently, adding other variants only requires small changes to VARIANTS_CONFIGS for each class, will be added asap
    variant = "s"
    
    backbone = YoloNASBackbone(
        in_channels=3,
        variant=variant
    )
    neck = YoloNASNeck(
        variant=variant
    )
    head = YoloNASPoseHead(
        num_classes=15,
        variant=variant
    )

    input = torch.randn(8,3,288,512)

    outputs = head(
        neck(
            backbone(
                input
            )
        )
    )

    predictions, raw = outputs

    for t in predictions:
        print(t.shape)