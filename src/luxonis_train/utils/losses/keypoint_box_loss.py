from typing import Any, Dict, List, Tuple, cast

import torch
from torch import Tensor, nn

from .base_loss import BaseLoss
from .box_loss import BoxLoss
from .common import BCEWithLogitsLoss, SmoothBCEWithLogitsLoss
from .keypoint_loss import KeypointLoss
from luxonis_train.utils.registry import LOSSES


@LOSSES.register_module()
class KeypointBoxLoss(BaseLoss):
    """Joint loss for keypoint and box predictions for cases where the keypoints and boxes are inherently linked.
    Based on `YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss`,
    https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf
    """

    def __init__(
        self,
        cls_pw: float = 1.0,
        obj_pw: float = 1.0,
        label_smoothing: float = 0.0,
        iou_ratio: float = 1,
        box_weight: float = 0.05,
        kpt_weight: float = 0.10,
        kptv_weight: float = 0.6,
        cls_weight: float = 0.6,
        obj_weight: float = 0.7,
        anchor_t: float = 4.0,
        bias: float = 0.5,
        balance: List[float] = [4.0, 1.0, 0.4],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_classes: int = cast(int, self.head_attributes.get("n_classes"))
        self.n_keypoints: int = cast(int, self.head_attributes.get("n_keypoints"))
        self.n_anchors: int = cast(int, self.head_attributes.get("n_anchors"))
        self.num_heads: int = cast(int, self.head_attributes.get("num_heads"))
        self.anchors: Tensor = cast(Tensor, self.head_attributes.get("anchors"))
        self.box_offset: int = cast(int, self.head_attributes.get("box_offset"))
        self.balance = balance
        if len(self.balance) < self.num_heads:
            raise ValueError(
                f"Balance list must have at least {self.num_heads} elements."
            )

        self.iou_ratio = iou_ratio
        self.box_weight = box_weight
        self.kpt_weight = kpt_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.kptv_weight = kptv_weight
        self.anchor_threshold = anchor_t

        self.bias = bias

        self.BCE = BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))
        self.class_loss = SmoothBCEWithLogitsLoss(
            label_smoothing=label_smoothing, bce_pow=cls_pw
        )
        self.keypoint_loss = KeypointLoss(bce_power=cls_pw)
        self.box_loss = BoxLoss()

        self.positive_smooth_const = 1 - 0.5 * label_smoothing
        self.negative_smooth_const = 0.5 * label_smoothing

    def forward(
        self,
        model_output: Tuple[Tensor, List[Tensor]],
        targets: Tuple[
            List[Tensor],
            List[Tensor],
            List[Tensor],
            List[Tuple[Tensor, Tensor, Tensor, Tensor]],
            List[Tensor],
        ],
        epoch: int,
        step: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        predictions = model_output[1]
        device = predictions[0].device
        sub_losses = {
            "box": torch.zeros(1, device=device),
            "obj": torch.zeros(1, device=device),
            "cls": torch.zeros(1, device=device),
            "kptv": torch.zeros(1, device=device),
            "kpt": torch.zeros(1, device=device),
        }

        for pred, class_target, box_target, kpt_target, index, anchor, balance in zip(
            predictions, *targets, self.balance
        ):
            obj_targets = torch.zeros_like(pred[..., 0], device=device)

            n_targets = len(class_target)
            if n_targets > 0:
                pred_subset = pred[index]

                box_loss, iou = self.box_loss(pred_subset[:, :4], anchor, box_target)
                sub_losses["box"] += box_loss * self.box_weight

                kpt_loss, kpt_visibility_loss = self.keypoint_loss(
                    pred_subset[:, self.box_offset + self.n_classes :],
                    kpt_target.to(device),
                    epoch,
                    step,
                )

                sub_losses["kpt"] += kpt_loss * self.kpt_weight
                sub_losses["kptv"] += kpt_visibility_loss * self.kptv_weight

                obj_targets[index] = (
                    1.0 - self.iou_ratio
                ) + self.iou_ratio * iou.detach().clamp(0).to(obj_targets.dtype)

                if self.n_classes > 1:
                    sub_losses["cls"] += (
                        self.class_loss(
                            pred_subset[
                                :, self.box_offset : self.box_offset + self.n_classes
                            ],
                            class_target,
                            epoch,
                            step,
                        )[0]
                        * self.cls_weight
                    )

            sub_losses["obj"] += (
                self.BCE(pred[..., 4], obj_targets, epoch, step)[0]
                * balance
                * self.obj_weight
            )

        loss = cast(Tensor, sum(sub_losses.values())).reshape([])
        return loss, {name: loss.detach() for name, loss in sub_losses.items()}
