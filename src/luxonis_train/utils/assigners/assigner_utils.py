import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.utils.boxutils import bbox_iou


def candidates_in_gt(
    anchor_centers: Tensor, gt_bboxes: Tensor, eps: float = 1e-9
) -> Tensor:
    """Check if anchor box's center is in any GT bbox.

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
    candidates = (bbox_delta.min(dim=-1)[0] > eps).to(gt_bboxes.dtype)
    return candidates


def fix_collisions(
    mask_pos: Tensor, overlaps: Tensor, n_max_boxes: int
) -> tuple[Tensor, Tensor, Tensor]:
    """If an anchor is assigned to multiple GTs, the one with highest IoU is selected.

    Args:
        mask_pos (Tensor): Mask of assigned anchors [bs, n_max_boxes, n_anchors]
        overlaps (Tensor): IoUs between GTs and anchors [bx, n_max_boxes, n_anchors]
        n_max_boxes (int): Number of maximum boxes per image

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Assigned indices, sum of positive mask, positive mask
    """
    mask_pos_sum = mask_pos.sum(dim=-2)
    if mask_pos_sum.max() > 1:
        mask_multi_gts = (mask_pos_sum.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(dim=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        mask_pos_sum = mask_pos.sum(dim=-2)
    assigned_gt_idx = mask_pos.argmax(dim=-2)
    return assigned_gt_idx, mask_pos_sum, mask_pos


def batch_iou(batch1: Tensor, batch2: Tensor) -> Tensor:
    """Calculates IoU for each pair of bboxes in the batch. Bboxes must be in xyxy
    format.

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
