#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/loss.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.ops import box_convert

from luxonis_train.utils.config import Config
from luxonis_train.utils.assigners import (
    ATSSAssigner,
    TaskAlignedAssigner,
    generate_anchors,
)
from luxonis_train.utils.boxutils import dist2bbox, bbox_iou


class YoloV6Loss(nn.Module):
    """Loss computation func."""

    def __init__(
        self,
        n_classes,
        image_size=None,
        fpn_strides=None,
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        warmup_epoch=4,
        iou_type="giou",
        loss_weight={"class": 1.0, "iou": 2.5},
        **kwargs,
    ):
        super(YoloV6Loss, self).__init__()

        is_4head = kwargs.get("head_attributes").get("is_4head", False)
        if fpn_strides is None:
            self.fpn_strides = [4, 8, 16, 32] if is_4head else [8, 16, 32]
        else:
            self.fpn_strides = fpn_strides

        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = n_classes

        # if no image size provided get it from config
        if image_size is None:
            cfg = Config()
            image_size = cfg.get("train.preprocessing.train_image_size")
        self.original_img_size = image_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(
            topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0
        )

        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss()
        self.bbox_loss = BboxLoss(self.num_classes, self.iou_type)
        self.loss_weight = loss_weight

    def forward(
        self,
        outputs,
        targets,
        **kwargs,
    ):
        epoch_num = kwargs["epoch"]
        step_num = kwargs["step"]

        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
            feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset
        )
        assert pred_scores.type() == pred_distri.type()

        # gt_bboxes_scale = torch.full((1,4), img_size).type_as(pred_scores)
        # supports rectangular original images
        gt_bboxes_scale = torch.tensor(
            [
                self.original_img_size[1],
                self.original_img_size[0],
                self.original_img_size[1],
                self.original_img_size[0],
            ],
            device=targets.device,
        )  # use if bboxes normalized
        batch_size = pred_scores.shape[0]

        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = dist2bbox(pred_distri, anchor_points_s)

        try:
            if epoch_num < self.warmup_epoch:
                (
                    target_labels,
                    target_bboxes,
                    target_scores,
                    fg_mask,
                ) = self.warmup_assigner(
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
                ) = self.formal_assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                )

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                (
                    target_labels,
                    target_bboxes,
                    target_scores,
                    fg_mask,
                ) = self.warmup_assigner(
                    _anchors,
                    _n_anchors_list,
                    _gt_labels,
                    _gt_bboxes,
                    _mask_gt,
                    _pred_bboxes * _stride_tensor,
                )

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                (
                    target_labels,
                    target_bboxes,
                    target_scores,
                    fg_mask,
                ) = self.formal_assigner(
                    _pred_scores,
                    _pred_bboxes * _stride_tensor,
                    _anchor_points,
                    _gt_labels,
                    _gt_bboxes,
                    _mask_gt,
                )

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()

        # Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(
            fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes)
        )
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # bbox loss
        loss_iou = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
        )

        loss = self.loss_weight["class"] * loss_cls + self.loss_weight["iou"] * loss_iou

        sub_losses = {"class": loss_cls.detach(), "iou": loss_iou.detach()}

        return loss, sub_losses

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
            np.array(
                list(
                    map(
                        lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)),
                        targets_list,
                    )
                )
            )[:, 1:, :]
        ).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = box_convert(batch_target, "xywh", "xyxy")
        return targets


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.amp.autocast(enabled=False, device_type=pred_score.device.type):
            loss = (
                F.binary_cross_entropy(
                    pred_score.float(), gt_score.float(), reduction="none"
                )
                * weight
            ).sum()
        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, iou_type="giou"):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format="xyxy", iou_type=iou_type, eps=1e-10)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
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
            loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
        else:
            loss_iou = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou


class IOUloss(nn.Module):
    """Calculate IoU loss."""

    def __init__(self, box_format="xywh", iou_type="ciou", reduction="none", eps=1e-7):
        """Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        super(IOUloss, self).__init__()
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def forward(self, box1, box2, **kwargs):
        """calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4]."""
        iou = bbox_iou(box1, box2, box_format=self.box_format, iou_type=self.iou_type)
        loss = 1.0 - iou

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
