#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/loss.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal, List, Dict, Tuple
from torchvision.ops import box_convert

from luxonis_train.utils.losses.base_loss import BaseLoss
from luxonis_train.utils.assigners import (
    ATSSAssigner,
    TaskAlignedAssigner,
)
from luxonis_train.utils.boxutils import anchors_for_fpn_features, dist2bbox, bbox_iou


class YoloV6Loss(BaseLoss):
    def __init__(
        self,
        n_warmup_epochs: int = 4,
        iou_type: Literal["none", "ciou", "diou", "giou", "siou"] = "giou",
        loss_weight: Dict[str, float] = {"class": 1.0, "iou": 2.5},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_classes: int = self.head_attributes.get("n_classes")
        self.stride: Tensor = self.head_attributes.get("stride")
        self.grid_cell_size: float = self.head_attributes.get("grid_cell_size")
        self.grid_cell_offset: float = self.head_attributes.get("grid_cell_offset")
        self.original_img_size: List[int] = self.head_attributes.get(
            "original_in_shape"
        )[
            2:
        ]  # take only [H,W]

        self.n_warmup_epochs = n_warmup_epochs
        self.atts_assigner = ATSSAssigner(topk=9, n_classes=self.n_classes)
        self.tal_assigner = TaskAlignedAssigner(
            topk=13, n_classes=self.n_classes, alpha=1.0, beta=6.0
        )

        self.varifocal_loss = VarifocalLoss()
        self.bbox_iou_loss = BboxIoULoss(iou_type)
        self.loss_weight = loss_weight

    def forward(
        self,
        preds: Tuple[List[Tensor], Tensor, Tensor],
        target: Tensor,
        epoch: int,
        step: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        feats, pred_scores, pred_distri = preds

        (
            anchors,
            anchor_points,
            n_anchors_list,
            stride_tensor,
        ) = anchors_for_fpn_features(
            feats,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=True,
        )

        # target preprocessing
        gt_bboxes_scale = torch.tensor(
            [
                self.original_img_size[1],
                self.original_img_size[0],
                self.original_img_size[1],
                self.original_img_size[0],
            ],
            device=target.device,
        )  # use if bboxes normalized
        batch_size = pred_scores.shape[0]
        target = self._preprocess_target(target, batch_size, gt_bboxes_scale)
        gt_labels = target[:, :, :1]  # cls
        gt_bboxes = target[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # preds preprocessing
        anchor_points_strided = anchor_points / stride_tensor
        pred_bboxes = dist2bbox(pred_distri, anchor_points_strided)

        if epoch < self.n_warmup_epochs:
            (
                target_labels,
                target_bboxes,
                target_scores,
                fg_mask,
            ) = self.atts_assigner(
                anchors,
                n_anchors_list,
                gt_labels,
                gt_bboxes,
                mask_gt,
                pred_bboxes.detach() * stride_tensor,
            )
        else:
            (
                target_labels,
                target_bboxes,
                target_scores,
                fg_mask,
            ) = self.tal_assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        # cls loss
        target_labels = torch.where(
            fg_mask > 0, target_labels, torch.full_like(target_labels, self.n_classes)
        )
        one_hot_label = F.one_hot(target_labels.long(), self.n_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # bbox iou loss
        loss_iou = self.bbox_iou_loss(
            pred_distri,
            pred_bboxes,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
        )

        loss = self.loss_weight["class"] * loss_cls + self.loss_weight["iou"] * loss_iou

        sub_losses = {"class": loss_cls.detach(), "iou": loss_iou.detach()}

        return loss, sub_losses

    def _preprocess_target(self, target: Tensor, batch_size: int, scale_tensor: Tensor):
        """Preprocess target in shape [batch_size, N, 5] where N is maximum number of instances in one image"""
        sample_ids, counts = torch.unique(target[:, 0].int(), return_counts=True)
        out_target = torch.zeros(batch_size, counts.max(), 5, device=target.device)
        out_target[:, :, 0] = -1
        for id, count in zip(sample_ids, counts):
            out_target[id, :count] = target[target[:, 0] == id][:, 1:]

        scaled_target = out_target[:, :, 1:5] * scale_tensor
        out_target[..., 1:] = box_convert(scaled_target, "xywh", "xyxy")
        return out_target


class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        """Varifocal Loss is a loss function for training a dense object detector to predict
        the IoU-aware classification score, inspired by focal loss.

        Args:
            alpha (float, optional): Defaults to 0.75.
            gamma (float, optional): Defaults to 2.0.
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score: Tensor, target_score: Tensor, label: Tensor):
        weight = (
            self.alpha * pred_score.pow(self.gamma) * (1 - label) + target_score * label
        )
        ce_loss = F.binary_cross_entropy(
            pred_score.float(), target_score.float(), reduction="none"
        )
        loss = (ce_loss * weight).sum()
        return loss


class BboxIoULoss(nn.Module):
    def __init__(self, iou_type: Literal["none", "giou", "ciou", "siou"] = "giou"):
        """IoU loss on bounding boxes

        Args:
            iou_type (Literal["none", "giou", "ciou", "siou"], optional): Defaults to "giou".
        """
        super().__init__()

        self.iou_type = iou_type

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: Tensor,
        fg_mask: Tensor,
    ):
        num_pos = fg_mask.sum()
        if num_pos > 0:
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape(
                [-1, 4]
            )
            target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).reshape(
                [-1, 4]
            )
            bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(
                -1
            )

            iou = bbox_iou(
                pred_bboxes_pos,
                target_bboxes_pos,
                iou_type=self.iou_type,
                element_wise=True,
            )
            iou = iou[..., None]
            loss_iou = (1 - iou) * bbox_weight

            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
        else:
            loss_iou = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou
