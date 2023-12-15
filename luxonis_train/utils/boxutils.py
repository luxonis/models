"""This module contains various utility functions for working with bounding boxes."""

import math
from typing import Literal, TypeAlias

import torch
from scipy.cluster.vq import kmeans
from torch import Tensor
from torchvision.ops import (
    batched_nms,
    box_convert,
    box_iou,
    distance_box_iou,
    generalized_box_iou,
)

from luxonis_train.utils.types import LabelType

IoUType: TypeAlias = Literal["none", "giou", "diou", "ciou", "siou"]
BBoxFormatType: TypeAlias = Literal["xyxy", "xywh", "cxcywh"]

__all__ = [
    "anchors_for_fpn_features",
    "anchors_from_dataset",
    "bbox2dist",
    "bbox_iou",
    "compute_iou_loss",
    "dist2bbox",
    "match_to_anchor",
    "non_max_suppression",
    "process_bbox_predictions",
    "process_keypoints_predictions",
]


def match_to_anchor(
    targets: Tensor,
    anchor: Tensor,
    xy_shifts: Tensor,
    scale_width: int,
    scale_height: int,
    n_keypoints: int,
    anchor_threshold: float,
    bias: float,
    box_offset: int = 5,
) -> tuple[Tensor, Tensor]:
    """Matches targets to anchors.

    1. Scales the targets to the size of the feature map
    2. Matches the targets to the anchor, filtering out targets whose aspect
        ratio is too far from the anchor's aspect ratio.

    Args:
        targets (Tensor): Targets in xyxy format
        anchor (Tensor): Anchor boxes
        xy_shifts (Tensor): Shifts in x and y direction
        scale_width (int): Width of the feature map
        scale_height (int): Height of the feature map
        n_keypoints (int): Number of keypoints
        anchor_threshold (float): Threshold for anchor filtering
        bias (float): Bias for anchor filtering
        box_offset (int, optional): Offset for box. Defaults to 5.

    Returns:
        tuple[Tensor, Tensor]: Scaled targets and shifts.
    """

    # The boxes and keypoints need to be scaled to the size of the features
    # First two indices are batch index and class label,
    # last index is anchor index. Those are not scaled.
    scale_length = 2 * n_keypoints + box_offset + 2
    scales = torch.ones(scale_length, device=targets.device)
    scales[2 : scale_length - 1] = torch.tensor(
        [scale_width, scale_height] * (n_keypoints + 2)
    )
    scaled_targets = targets * scales
    if targets.size(1) == 0:
        return targets[0], torch.zeros(1, device=targets.device)

    wh_to_anchor_ratio = scaled_targets[:, :, 4:6] / anchor.unsqueeze(1)
    ratio_mask = (
        torch.max(wh_to_anchor_ratio, 1.0 / wh_to_anchor_ratio).max(2)[0]
        < anchor_threshold
    )

    filtered_targets = scaled_targets[ratio_mask]

    box_xy = filtered_targets[:, 2:4]
    box_wh = torch.tensor([scale_width, scale_height]) - box_xy

    def decimal_part(x: Tensor) -> Tensor:
        return x % 1.0

    x, y = ((decimal_part(box_xy) < bias) & (box_xy > 1.0)).T
    w, h = ((decimal_part(box_wh) < bias) & (box_wh > 1.0)).T
    mask = torch.stack((torch.ones_like(x), x, y, w, h))
    final_targets = filtered_targets.repeat((len(xy_shifts), 1, 1))[mask]

    shifts = xy_shifts.unsqueeze(1).repeat((1, len(box_xy), 1))[mask]
    return final_targets, shifts


def dist2bbox(
    distance: Tensor,
    anchor_points: Tensor,
    out_format: BBoxFormatType = "xyxy",
) -> Tensor:
    """Transform distance (ltrb) to box ("xyxy", "xywh" or "cxcywh").

    Args:
        distance (Tensor): Distance predictions
        anchor_points (Tensor): Head's anchor points
        out_format (Literal["xyxy", "xywh", "cxcywh"], optional): BBox output format. Defaults to "xyxy".

    Returns:
        Tensor: BBoxes in correct format
    """
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    bbox = torch.cat([x1y1, x2y2], -1)
    if out_format in ["xyxy", "xywh", "cxcywh"]:
        bbox = box_convert(bbox, in_fmt="xyxy", out_fmt=out_format)
    else:
        raise ValueError(f"Out format `{out_format}` for bbox not supported")
    return bbox


def bbox2dist(bbox: Tensor, anchor_points: Tensor, reg_max: float) -> Tensor:
    """Transform bbox(xyxy) to distance(ltrb).

    Args:
        bbox (Tensor): Bboxes in "xyxy" format
        anchor_points (Tensor): Head's anchor points
        reg_max (float): Maximum regression distances

    Returns:
        Tensor: BBoxes in distance(ltrb) format
    """
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist


def bbox_iou(
    bbox1: Tensor,
    bbox2: Tensor,
    bbox_format: BBoxFormatType = "xyxy",
    iou_type: IoUType = "none",
    element_wise: bool = False,
) -> Tensor:
    """Computes IoU between two sets of bounding boxes.

    Args:
        bbox1 (Tensor): First set of bboxes [N, 4]
        bbox2 (Tensor): Second set of bboxes [M, 4]
        box_format (Literal["xyxy", "xywh", "cxcywh"], optional): Input bbox format.
          Defaults to "xyxy".
        iou_type (Literal["none", "giou", "diou", "ciou", "siou"], optional): IoU type.
          Defaults to "none".
        element_wise (bool, optional): If True returns element wise IoUs.
          Defaults to False.

    Returns:
        Tensor: [N,M] or [N] tensor
    """
    if bbox_format != "xyxy":
        bbox1 = box_convert(bbox1, in_fmt=bbox_format, out_fmt="xyxy")
        bbox2 = box_convert(bbox2, in_fmt=bbox_format, out_fmt="xyxy")

    if iou_type == "none":
        iou = box_iou(bbox1, bbox2)
    elif iou_type == "giou":
        iou = generalized_box_iou(bbox1, bbox2)
    elif iou_type == "diou":
        iou = distance_box_iou(bbox1, bbox2)
    elif iou_type == "ciou":
        # CIoU from `Enhancing Geometric Factors in Model Learning and Inference for
        # Object Detection and Instance Segmentation`, https://arxiv.org/pdf/2005.03572.pdf.
        # Implementation adapted from torchvision complete_box_iou with added eps for stability
        eps = 1e-7

        iou = bbox_iou(bbox1, bbox2, iou_type="none")
        diou = bbox_iou(bbox1, bbox2, iou_type="diou")

        w1 = bbox1[:, None, 2] - bbox1[:, None, 0]
        h1 = bbox1[:, None, 3] - bbox1[:, None, 1] + eps
        w2 = bbox2[:, 2] - bbox2[:, 0]
        h2 = bbox2[:, 3] - bbox2[:, 1] + eps

        v = (4 / (torch.pi**2)) * torch.pow(
            torch.atan(w1 / h1) - torch.atan(w2 / h2), 2
        )
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        iou = diou - alpha * v

    elif iou_type == "siou":
        # SIoU from `SIoU Loss: More Powerful Learning for Bounding Box Regression`,
        # https://arxiv.org/pdf/2205.12740.pdf

        eps = 1e-7
        bbox1_xywh = box_convert(bbox1, in_fmt="xyxy", out_fmt="xywh")
        w1, h1 = bbox1_xywh[:, 2], bbox1_xywh[:, 3]
        bbox2_xywh = box_convert(bbox2, in_fmt="xyxy", out_fmt="xywh")
        w2, h2 = bbox2_xywh[:, 2], bbox2_xywh[:, 3]

        # enclose area
        enclose_x1y1 = torch.min(bbox1[:, None, :2], bbox2[:, :2])
        enclose_x2y2 = torch.max(bbox1[:, None, 2:], bbox2[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)
        cw = enclose_wh[..., 0]
        ch = enclose_wh[..., 1]

        # angle cost
        s_cw = (
            bbox2[:, None, 0] + bbox2[:, None, 2] - bbox1[:, 0] - bbox1[:, 2]
        ) * 0.5 + eps
        s_ch = (
            bbox2[:, None, 1] + bbox2[:, None, 3] - bbox1[:, 1] - bbox1[:, 3]
        ) * 0.5 + eps

        sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # distance cost
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        # shape cost
        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omega_w), 4) + torch.pow(
            1 - torch.exp(-1 * omega_h), 4
        )

        iou = box_iou(bbox1, bbox2) - 0.5 * (distance_cost + shape_cost)
    else:
        raise ValueError(f"IoU type `{iou_type}` not supported.")

    iou = torch.nan_to_num(iou, 0)

    if element_wise:
        return iou.diag()
    else:
        return iou


def non_max_suppression(
    preds: Tensor,
    n_classes: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    keep_classes: list[int] | None = None,
    agnostic: bool = False,
    multi_label: bool = False,
    bbox_format: BBoxFormatType = "xyxy",
    max_det: int = 300,
    predicts_objectness: bool = True,
) -> list[Tensor]:
    """Non-maximum suppression on model's predictions to keep only best instances.

    Args:
        preds (Tensor): Model's prediction tensor of shape [bs, N, M]
        n_classes (int): Number of model's classes
        conf_thres (float, optional): Boxes with confidence higher than this will be kept. Defaults to 0.25.
        iou_thres (float, optional): Boxes with IoU higher than this will be discarded. Defaults to 0.45.
        keep_classes (list[int] | None, optional): Subset of classes to keep,
            if None then keep all of them. Defaults to None.
        agnostic (bool, optional): Whether perform NMS per class or treat all classes
            the same. Defaults to False.
        multi_label (bool, optional): Whether one prediction can have multiple labels. Defaults to False.
        box_format (Literal["xyxy", "xywh", "cxcywh"], optional): Input bbox format. Defaults to "xyxy".
        max_det (int, optional): Number of maximum output detections. Defaults to 300.
        max_wh (int, optional): Maximum width and height of the bbox. Defaults to 4096.
        predicts_objectess (bool, optional): Whether head predicts objectness confidence. Defaults to True.

    Returns:
        list[Tensor]: list of kept detections for each image, boxes in "xyxy" format. Tensors with shape [n_kept, M]
    """
    if not (0 <= conf_thres <= 1):
        raise ValueError(
            f"Confidence threshold must be in range [0,1] but set to {conf_thres}."
        )
    if not (0 <= iou_thres <= 1):
        raise ValueError(
            f"IoU threshold must be in range [0,1] but set to {iou_thres}."
        )

    multi_label &= n_classes > 1

    # If any data after bboxes are present.
    has_additional = preds.size(-1) > (4 + 1 + n_classes)

    candidate_mask = preds[..., 4] > conf_thres
    if not predicts_objectness:
        candidate_mask = torch.logical_and(
            candidate_mask,
            torch.max(preds[..., 5 : 5 + n_classes], dim=-1)[0] > conf_thres,
        )

    output = [torch.zeros((0, preds.size(-1)), device=preds.device)] * preds.size(0)

    for i, x in enumerate(preds):
        curr_out = x[candidate_mask[i]]

        if curr_out.size(0) == 0:
            continue

        if predicts_objectness:
            if n_classes == 1:
                curr_out[:, 5 : 5 + n_classes] = curr_out[:, 4:5]
            else:
                curr_out[:, 5 : 5 + n_classes] *= curr_out[:, 4:5]
        else:
            curr_out[:, 5 : 5 + n_classes] *= curr_out[:, 4:5]

        bboxes = curr_out[:, :4]
        keep_mask = torch.zeros(bboxes.size(0)).bool()
        if bbox_format != "xyxy":
            bboxes = box_convert(bboxes, in_fmt=bbox_format, out_fmt="xyxy")

        if multi_label:
            box_idx, class_idx = (
                (curr_out[:, 5 : 5 + n_classes] > conf_thres).nonzero(as_tuple=False).T
            )
            keep_mask[box_idx] = True
            curr_out = torch.cat(
                (
                    bboxes[keep_mask],
                    curr_out[keep_mask, class_idx + 5, None],
                    class_idx[:, None].float(),
                ),
                1,
            )
        else:
            conf, class_idx = curr_out[:, 5 : 5 + n_classes].max(1, keepdim=True)
            keep_mask[conf.view(-1) > conf_thres] = True
            curr_out = torch.cat((bboxes, conf, class_idx.float()), 1)[keep_mask]

        if has_additional:
            curr_out = torch.hstack(
                [curr_out, x[candidate_mask[i]][keep_mask, 5 + n_classes :]]
            )

        if keep_classes is not None:
            curr_out = curr_out[
                (
                    curr_out[:, 5:6]
                    == torch.tensor(keep_classes, device=curr_out.device)
                ).any(1)
            ]

        if not curr_out.size(0):
            continue

        keep_indices = batched_nms(
            boxes=curr_out[:, :4],
            scores=curr_out[:, 4],
            iou_threshold=iou_thres,
            idxs=curr_out[:, 5].int() * (0 if agnostic else 1),
        )
        keep_indices = keep_indices[:max_det]

        output[i] = curr_out[keep_indices]

    return output


def anchors_from_dataset(
    loader: torch.utils.data.DataLoader,
    n_anchors: int = 9,
    n_generations: int = 1000,
    ratio_threshold: float = 4.0,
) -> tuple[Tensor, float]:
    """Generates anchors based on bounding box annotations present in provided data
    loader. It uses K-Means for initial proposals which are then refined with genetic
    algorithm.

    Args:
        loader (torch.utils.data.DataLoader): Data loader
        n_anchors (int, optional): Number of anchors, this is normally num_heads * 3 which generates 3 anchors per layer. Defaults to 9.
        n_generations (int, optional): Number of iterations for anchor improvement with genetic algorithm. Defaults to 1000.
        ratio_threshold (float, optional): Minimum threshold for ratio. Defaults to 4.0.

    Returns:
        tuple[Tensor, float]: Proposed anchors and the best possible recall.
    """

    widths = []
    inputs = None
    for inp, labels in loader:
        boxes = labels[LabelType.BOUNDINGBOX]
        curr_wh = boxes[:, 4:]
        widths.append(curr_wh)
        inputs = inp
    assert inputs is not None, "No inputs found in data loader"
    _, _, h, w = inputs.shape  # assuming all images are same size
    img_size = torch.tensor([w, h])
    wh = torch.vstack(widths) * img_size

    # filter out small objects (w or h < 2 pixels)
    wh = wh[(wh >= 2).any(1)]

    try:
        assert n_anchors <= len(
            wh
        ), "More requested anchors than number of bounding boxes."
        std = wh.std(0)
        proposed_anchors = kmeans(wh / std, n_anchors, iter=30)
        proposed_anchors = torch.tensor(proposed_anchors[0]) * std
        assert n_anchors == len(
            proposed_anchors
        ), "KMeans returned insufficient number of points"
    except Exception:
        print("Fallback to random anchor init")
        proposed_anchors = (
            torch.sort(torch.rand(n_anchors * 2))[0].reshape(n_anchors, 2) * img_size
        )

    proposed_anchors = proposed_anchors[
        torch.argsort(proposed_anchors.prod(1))
    ]  # sort small to large

    def calc_best_anchor_ratio(anchors: Tensor, wh: Tensor) -> Tensor:
        """Calculate how well most suitable anchor box matches each target bbox."""
        symmetric_size_ratios = torch.min(
            wh[:, None] / anchors[None], anchors[None] / wh[:, None]
        )
        worst_side_size_ratio = symmetric_size_ratios.min(-1).values
        best_anchor_ratio = worst_side_size_ratio.max(-1).values
        return best_anchor_ratio

    def calc_best_possible_recall(anchors: Tensor, wh: Tensor) -> Tensor:
        """Calculate best possible recall if every bbox is matched to an appropriate
        anchor."""
        best_anchor_ratio = calc_best_anchor_ratio(anchors, wh)
        best_possible_recall = (best_anchor_ratio > 1 / ratio_threshold).float().mean()
        return best_possible_recall

    def anchor_fitness(anchors: Tensor, wh: Tensor) -> Tensor:
        """Fitness function used for anchor evolve."""
        best_anchor_ratio = calc_best_anchor_ratio(anchors, wh)
        return (
            best_anchor_ratio * (best_anchor_ratio > 1 / ratio_threshold).float()
        ).mean()

    # Genetic algorithm
    best_fitness = anchor_fitness(proposed_anchors, wh)
    anchor_shape = proposed_anchors.shape
    mutation_probability = 0.9
    mutation_noise_mean = 1
    mutation_noise_std = 0.1
    for _ in range(n_generations):
        anchor_mutation = torch.ones(anchor_shape)
        anchor_mutation = (
            (torch.rand(anchor_shape) < mutation_probability)
            * torch.randn(anchor_shape)
            * mutation_noise_std
            + mutation_noise_mean
        ).clip(0.3, 3.0)

        mutated_anchors = (proposed_anchors.clone() * anchor_mutation).clip(min=2.0)
        mutated_fitness = anchor_fitness(mutated_anchors, wh)
        if mutated_fitness > best_fitness:
            best_fitness = mutated_fitness
            proposed_anchors = mutated_anchors.clone()

    proposed_anchors = proposed_anchors[
        torch.argsort(proposed_anchors.prod(1))
    ]  # sort small to large
    recall = calc_best_possible_recall(proposed_anchors, wh)

    return proposed_anchors, recall.item()


def anchors_for_fpn_features(
    features: list[Tensor],
    strides: Tensor,
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    multiply_with_stride: bool = False,
) -> tuple[Tensor, Tensor, list[int], Tensor]:
    """Generates anchor boxes, points and strides based on FPN feature shapes and
    strides.

    Args:
        features (list[Tensor]): list of FPN features
        strides (Tensor): Strides of FPN features
        grid_cell_size (float, optional): Cell size in respect to input image size. Defaults to 5.0.
        grid_cell_offset (float, optional): Percent grid cell center's offset. Defaults to 0.5.
        multiply_with_stride (bool, optional): Whether to multiply per FPN values with its stride. Defaults to False.

    Returns:
        tuple[Tensor, Tensor, list[int], Tensor]: BBox anchors, center anchors, number of anchors, strides
    """
    anchors: list[Tensor] = []
    anchor_points: list[Tensor] = []
    n_anchors_list: list[int] = []
    stride_tensor: list[Tensor] = []
    for feature, stride in zip(features, strides):
        _, _, h, w = feature.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = torch.arange(end=w) + grid_cell_offset
        shift_y = torch.arange(end=h) + grid_cell_offset
        if multiply_with_stride:
            shift_x *= stride
            shift_y *= stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        anchor = (
            torch.stack(
                [
                    shift_x - cell_half_size,
                    shift_y - cell_half_size,
                    shift_x + cell_half_size,
                    shift_y + cell_half_size,
                ],
                dim=-1,
            )
            .reshape(-1, 4)
            .to(feature.dtype)
        )
        anchors.append(anchor)

        anchor_point = (
            torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2).to(feature.dtype)
        )
        anchor_points.append(anchor_point)

        curr_n_anchors = len(anchor)
        n_anchors_list.append(curr_n_anchors)
        stride_tensor.append(
            torch.full((curr_n_anchors, 1), stride, dtype=feature.dtype)  # type: ignore
        )

    device = features[0].device
    return (
        torch.cat(anchors).to(device),
        torch.cat(anchor_points).to(device),
        n_anchors_list,
        torch.cat(stride_tensor).to(device),
    )


def process_keypoints_predictions(keypoints: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Extracts x, y and visibility from keypoints predictions.

    Args:
        keypoints (Tensor[N, M]): Keypoints predictions. The last dimension must be
          divisible by 3 and is expected to be in format [x1, y1, v1, x2, y2, v2, ...].

    Returns:
        tuple[Tensor, Tensor, Tensor]: x, y and visibility tensors.
    """
    x = keypoints[..., ::3] * 2.0 - 0.5
    y = keypoints[..., 1::3] * 2.0 - 0.5
    visibility = keypoints[..., 2::3]
    return (
        x,
        y,
        visibility,
    )


def process_bbox_predictions(
    bbox: Tensor, anchor: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Transforms bbox predictions to correct format.

    Args:
        bbox (Tensor[N, 4 + ...]): Bbox predictions
        anchor (Tensor): Anchor boxes

    Returns:
        tuple[Tensor, Tensor, Tensor]: xy and wh predictions and tail. The tail is
          anything after xywh.
    """
    out_bbox = bbox.sigmoid()
    out_bbox_xy = out_bbox[..., 0:2] * 2.0 - 0.5
    out_bbox_wh = (out_bbox[..., 2:4] * 2) ** 2 * anchor
    out_bbox_tail = out_bbox[..., 4:]
    return out_bbox_xy, out_bbox_wh, out_bbox_tail


def compute_iou_loss(
    pred_bboxes: Tensor,
    target_bboxes: Tensor,
    target_scores: Tensor | None = None,
    mask_positive: Tensor | None = None,
    *,
    iou_type: IoUType = "giou",
    bbox_format: BBoxFormatType = "xyxy",
    reduction: Literal["sum", "mean"] = "mean",
) -> tuple[Tensor, Tensor]:
    device = pred_bboxes.device
    target_bboxes = target_bboxes.to(device)
    if mask_positive is None or mask_positive.sum() > 0:
        if target_scores is not None:
            bbox_weight = torch.masked_select(
                target_scores.sum(-1),
                mask_positive
                if mask_positive is not None
                else torch.ones_like(target_scores.sum(-1)),
            ).unsqueeze(-1)
        else:
            bbox_weight = torch.tensor(1.0)

        if mask_positive is not None:
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
        else:
            bbox_mask = torch.ones_like(pred_bboxes, dtype=torch.bool)

        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).reshape(
            [-1, 4]
        )

        iou = bbox_iou(
            pred_bboxes_pos,
            target_bboxes_pos,
            iou_type=iou_type,
            bbox_format=bbox_format,
            element_wise=True,
        ).unsqueeze(-1)
        loss_iou = (1 - iou) * bbox_weight

        if reduction == "mean":
            loss_iou = loss_iou.mean()

        elif reduction == "sum":
            if target_scores is None:
                raise NotImplementedError(
                    "Sum reduction is not supported when `target_scores` is None"
                )
            loss_iou = loss_iou.sum()
            if target_scores.sum() > 1:
                loss_iou /= target_scores.sum()
        else:
            raise ValueError(f"Unknown reduction type `{reduction}`")
    else:
        loss_iou = torch.tensor(0.0).to(pred_bboxes.device)
        iou = torch.zeros([len(target_bboxes)]).to(pred_bboxes.device)

    return loss_iou, iou.detach().clamp(0)