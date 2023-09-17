import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Tuple

from luxonis_train.utils.boxutils import bbox_iou


class ATSSAssigner(nn.Module):
    def __init__(self, topk: int = 9, n_classes: int = 80):
        """Adaptive Training Sample Selection Assigner from `Bridging the Gap Between Anchor-based and Anchor-free Detection via
        Adaptive Training Sample Selection`, https://arxiv.org/pdf/1912.02424.pdf.
        Code is adapted from: https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/atss_assigner.py and
        https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            topk (int, optional): Number of anchors considere in selection. Defaults to 9.
            n_classes (int, optional): Number of classes in the dataset. Defaults to 80.
        """
        super().__init__()

        self.topk = topk
        self.n_classes = n_classes

    def forward(
        self,
        anchor_bboxes: Tensor,
        n_level_bboxes: List[int],
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
        pred_bboxes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assigner's forward method which generates final assignments

        Args:
            anchor_bboxes (Tensor): Anchor bboxes of shape [n_anchors, 4]
            n_level_bboxes (List[int]): Number of bboxes per level
            gt_labels (Tensor): Initial GT labels [bs, n_max_boxes, 1]
            gt_bboxes (Tensor): Initial GT bboxes [bs, n_max_boxes, 4]
            mask_gt (Tensor): Mask for valid GTs [bs, n_max_boxes, 1]
            pred_bboxes (Tensor): Predicted bboxes of shape [bs, n_anchors, 4]

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Assigned labels of shape [bs, n_anchors],
                assigned bboxes of shape [bs, n_anchors, 4], assigned scores of shape [bs, n_anchors, 1]
                and output positive mask of shape [bs, n_anchors]
        """

        self.n_anchors = anchor_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full([self.bs, self.n_anchors], self.n_classes).to(device),
                torch.zeros([self.bs, self.n_anchors, 4]).to(device),
                torch.zeros([self.bs, self.n_anchors, self.n_classes]).to(device),
                torch.zeros([self.bs, self.n_anchors]).to(device),
            )

        gt_bboxes_flat = gt_bboxes.reshape([-1, 4])

        # Compute iou between all gt and anchor bboxes
        overlaps = bbox_iou(gt_bboxes_flat, anchor_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        # Compute center distance between all gt and anchor bboxes
        gt_centers = self._get_bbox_center(gt_bboxes_flat)
        anchor_centers = self._get_bbox_center(anchor_bboxes)
        distances = (
            (gt_centers[:, None, :] - anchor_centers[None, :, :]).pow(2).sum(-1).sqrt()
        )
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        # Select candidates based on the center distance
        is_in_topk, topk_idxs = self._select_topk_candidates(
            distances, n_level_bboxes, mask_gt
        )

        # Compute threshold and selected positive candidates based on it
        is_pos = self._get_positive_samples(is_in_topk, topk_idxs, overlaps)

        # Select candidates inside GT
        is_in_gts = candidates_in_gt(anchor_centers, gt_bboxes_flat)
        is_in_gts = torch.reshape(is_in_gts, (self.bs, self.n_max_boxes, -1))

        # Final positive candidates
        mask_pos = is_pos * is_in_gts * mask_gt

        # If an anchor box is assigned to multiple gts, the one with the highest IoU is selected
        assigned_gt_idx, mask_pos_sum, mask_pos = fix_collisions(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Generate final assignments based on masks
        assigned_labels, assigned_bboxes, assigned_scores = self._get_final_assignments(
            gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
        )

        # Soft label with IoU
        if pred_bboxes is not None:
            ious = batch_iou(gt_bboxes, pred_bboxes) * mask_pos
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            assigned_scores *= ious

        out_mask_positive = mask_pos_sum.bool()

        return (
            assigned_labels.long(),
            assigned_bboxes,
            assigned_scores,
            out_mask_positive,
        )

    def _get_bbox_center(self, bbox: Tensor) -> Tensor:
        """Computes centers of bbox with shape [N,4]"""
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        return torch.stack((cx, cy), dim=1).to(bbox.device)

    def _select_topk_candidates(
        self, distances: Tensor, n_level_bboxes: List[int], mask_gt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Select k anchors whose centers are closest to GT

        Args:
            distances (Tensor): Distances between GT and anchor centers
            n_level_bboxes (List[int]): List of number of bboxes per level
            mask_gt (Tensor): Mask for valid GT per image
        """
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_topk_list = []
        topk_idxs = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(
            level_distances, n_level_bboxes
        ):
            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(
                selected_k, dim=-1, largest=False
            )
            topk_idxs.append(per_level_topk_idxs + start_idx)
            per_level_topk_idxs = torch.where(
                mask_gt, per_level_topk_idxs, torch.zeros_like(per_level_topk_idxs)
            )
            is_in_topk = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            is_in_topk = torch.where(
                is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk
            )
            is_in_topk_list.append(is_in_topk.to(distances.dtype))
            start_idx = end_idx

        is_in_topk_list = torch.cat(is_in_topk_list, dim=-1)
        topk_idxs = torch.cat(topk_idxs, dim=-1)
        return is_in_topk_list, topk_idxs

    def _get_positive_samples(
        self,
        is_in_topk: Tensor,
        topk_idxs: Tensor,
        overlaps: Tensor,
    ) -> Tensor:
        """Computes threshold and returns mask for samples over threshold

        Args:
            is_in_topk (Tensor): Mask of candidate samples [bx, n_max_boxes, n_anchors]
            topk_idxs (Tensor): Indices of candidates [bx, n_max_boxes, topK*n_levels]
            overlaps (Tensor): IoUs between GTs and anchors [bx, n_max_boxes, n_anchors]
        """
        n_bs_max_boxes = self.bs * self.n_max_boxes
        _candidate_overlaps = torch.where(
            is_in_topk > 0, overlaps, torch.zeros_like(overlaps)
        )
        topk_idxs = topk_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * torch.arange(
            n_bs_max_boxes, device=topk_idxs.device
        )
        assist_idxs = assist_idxs[:, None]
        flatten_idxs = topk_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_idxs]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = torch.where(
            _candidate_overlaps > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_topk,
            torch.zeros_like(is_in_topk),
        )
        return is_pos

    def _get_final_assignments(
        self,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        assigned_gt_idx: Tensor,
        mask_pos_sum: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate final assignments based on the mask

        Args:
            gt_labels (Tensor): Initial GT labels [bs, n_max_boxes, 1]
            gt_bboxes (Tensor): Initial GT bboxes [bs, n_max_boxes, 4]
            assigned_gt_idx (Tensor): Indices of matched GTs [bs, n_anchors]
            mask_pos_sum (Tensor): Mask of matched GTs [bs, n_anchors]
        """
        # assigned target labels
        batch_idx = torch.arange(
            self.bs, dtype=gt_labels.dtype, device=gt_labels.device
        )
        batch_idx = batch_idx[..., None]
        assigned_gt_idx = (assigned_gt_idx + batch_idx * self.n_max_boxes).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_idx.flatten()]
        assigned_labels = assigned_labels.reshape([self.bs, self.n_anchors])
        assigned_labels = torch.where(
            mask_pos_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, self.n_classes),
        )

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idx.flatten()]
        assigned_bboxes = assigned_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores
        assigned_scores = F.one_hot(assigned_labels.long(), self.n_classes + 1).float()
        assigned_scores = assigned_scores[:, :, : self.n_classes]

        return assigned_labels, assigned_bboxes, assigned_scores


class TaskAlignedAssigner(nn.Module):
    def __init__(
        self,
        topk: int = 13,
        n_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        """Task Aligned Assigner from `TOOD: Task-aligned One-stage Object Detection`, https://arxiv.org/pdf/2108.07755.pdf
        Cose is adapted from: https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            topk (int, optional): Number of anchors considere in selection. Defaults to 13.
            n_classes (int, optional): Number of classes in the dataset. Defaults to 80.
            alpha (float, optional): Defaults to 1.0.
            beta (float, optional): Defaults to 6.0.
            eps (float, optional): Defaults to 1e-9.
        """
        super().__init__()

        self.topk = topk
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assigner's forward method which generates final assignments

        Args:
            pred_scores (Tensor): Predicted scores [bs, n_achors, 1]
            pred_bboxes (Tensor): Predicted bboxes [bs, n_acnhors, 4]
            anchor_points (Tensor): Anchor points [n_acnhors, 2]
            gt_labels (Tensor): Initial GT labels [bs, n_max_boxes, 1]
            gt_bboxes (Tensor): Initial GT bboxes [bs, n_max_boxes, 4]
            mask_gt (Tensor): Mask for valid GTs [bs, n_max_boxes, 1]

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Assigned labels of shape [bs, n_anchors],
                assigned bboxes of shape [bs, n_anchors, 4], assigned scores of shape [bs, n_anchors, 1]
                and output mask of shape [bs, n_anchors]
        """
        self.bs = pred_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pred_scores[..., 0], self.n_classes).to(device),
                torch.zeros_like(pred_bboxes).to(device),
                torch.zeros_like(pred_scores).to(device),
                torch.zeros_like(pred_scores[..., 0]).to(device),
            )

        # Compute alignment metric between all bboxes (bboxes of all pyramid levels) and GT
        align_metric, overlaps = self._get_alignment_metric(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes
        )

        # Select top-k bboxes as candidates for each GT
        is_in_gts = candidates_in_gt(anchor_points, gt_bboxes.reshape([-1, 4]))
        is_in_gts = torch.reshape(is_in_gts, (self.bs, self.n_max_boxes, -1))
        is_in_topk = self._select_topk_candidates(
            align_metric * is_in_gts,
            topk_mask=mask_gt.repeat([1, 1, self.topk]).bool(),
        )

        # Final positive candidates
        mask_pos = is_in_topk * is_in_gts * mask_gt

        # If an anchor box is assigned to multiple gts, the one with the highest IoU is selected
        assigned_gt_idx, mask_pos_sum, mask_pos = fix_collisions(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Generate final targets based on masks
        assigned_labels, assigned_bboxes, assigned_scores = self._get_final_assignments(
            gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
        )

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .max(-2)[0]
            .unsqueeze(-1)
        )
        assigned_scores = assigned_scores * norm_align_metric

        out_mask_positive = mask_pos_sum.bool()

        return assigned_labels, assigned_bboxes, assigned_scores, out_mask_positive

    def _get_alignment_metric(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
    ):
        """Calculates anchor alignment metric and IoU between GTs and predicted bboxes

        Args:
            pred_scores (Tensor): Tensor of shape [bs, n_anchors, 1]
            pred_bboxes (Tensor): Tensor of shape [bs, n_anchors, 4]
            gt_labels (Tensor): Tensor of shape [bs, n_max_boxes, 1]
            gt_bboxes (Tensor): Tensor of shape [bs, n_max_boxes, 4]
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[ind[0], ind[1]]

        overlaps = batch_iou(gt_bboxes, pred_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def _select_topk_candidates(
        self,
        metrics: Tensor,
        largest: bool = True,
        topk_mask: Optional[Tensor] = None,
    ):
        """Selects k anchors based on provided metrics tensor

        Args:
            metrics (Tensor): Metrics tensor of shape [bs, n_max_boxes, n_anchors]
            largest (bool, optional): Flag if should keep largest topK. Defaults to True.
            topk_mask (Optional[Tensor], optional): Mask for valid GTs of shape [bs, n_max_boxes, topk]. Defaults to None.

        """
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, axis=-1, largest=largest
        )
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile(
                [1, 1, self.topk]
            )
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(
            is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk
        )
        return is_in_topk.to(metrics.dtype)

    def _get_final_assignments(
        self,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        assigned_gt_idx: Tensor,
        mask_pos_sum: Tensor,
    ):
        """Generate final assignments based on the mask

        Args:
            gt_labels (Tensor): Initial GT labels [bs, n_max_boxes, 1]
            gt_bboxes (Tensor): Initial GT bboxes [bs, n_max_boxes, 4]
            assigned_gt_idx (Tensor): Indices of matched GTs [bs, n_anchors]
            mask_pos_sum (Tensor): Mask of matched GTs [bs, n_anchors]
        """
        # assigned target labels
        batch_ind = torch.arange(
            end=self.bs, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        assigned_gt_idx = assigned_gt_idx + batch_ind * self.n_max_boxes
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idx]

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idx]

        # assigned target scores
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.n_classes)
        mask_pos_scores = mask_pos_sum[:, :, None].repeat(1, 1, self.n_classes)
        assigned_scores = torch.where(
            mask_pos_scores > 0, assigned_scores, torch.full_like(assigned_scores, 0)
        )

        return assigned_labels, assigned_bboxes, assigned_scores


def candidates_in_gt(
    anchor_centers: Tensor, gt_bboxes: Tensor, eps: float = 1e-9
) -> Tensor:
    """Check if anchor box's center is in any GT bbox

    Args:
        anchor_centers (Tensor): Centers of anchor bboxes [n_anchors, 2]
        gt_bboxes (Tensor): Ground truth bboxes [bs * n_max_boxes, 4]
        eps (float, optional): Threshold for minimum delta. Defaults to 1e-9.

    Returns:
        Tensor: Mask for anchors inside any GT bbox
    """
    n_anchors = anchor_centers.size(0)
    anchor_centers = anchor_centers.unsqueeze(0).repeat(gt_bboxes.size(0), 1, 1)
    gt_bboxes_lt = gt_bboxes[:, :2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:].unsqueeze(1).repeat(1, n_anchors, 1)
    bbox_delta_lt = anchor_centers - gt_bboxes_lt
    bbox_delta_rb = gt_bboxes_rb - anchor_centers
    bbox_delta = torch.cat([bbox_delta_lt, bbox_delta_rb], dim=-1)
    candidates = (bbox_delta.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)
    return candidates


def fix_collisions(
    mask_pos: Tensor, overlaps: Tensor, n_max_boxes: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """If an anchor is assigned to multiple GTs, the one with highest IoU is selected

    Args:
        mask_pos (Tensor): Mask of assigned anchors [bs, n_max_boxes, n_anchors]
        overlaps (Tensor): IoUs between GTs and anchors [bx, n_max_boxes, n_anchors]
        n_max_boxes (int): Number of maximum boxes per image

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Assigned indices, sum of positive mask, positive mask
    """
    mask_pos_sum = mask_pos.sum(axis=-2)
    if mask_pos_sum.max() > 1:
        mask_multi_gts = (mask_pos_sum.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        mask_pos_sum = mask_pos.sum(axis=-2)
    assigned_gt_idx = mask_pos.argmax(axis=-2)
    return assigned_gt_idx, mask_pos_sum, mask_pos


def batch_iou(batch1: Tensor, batch2: Tensor) -> Tensor:
    """Calculates IoU for each pair of bboxes in the batch.
    Bboxes must be in xyxy format.

    Args:
        batch1 (Tensor): Tensor of shape [bs, N, 4]
        batch2 (Tensor): Tensor of shape [bs, M, 4]

    Returns:
        Tensor: Per image box IoU of shape [bs, N, M]
    """
    ious = torch.stack(
        [bbox_iou(batch1[i], batch2[i]) for i in range(batch1.size(0))], dim=0
    )
    return ious
