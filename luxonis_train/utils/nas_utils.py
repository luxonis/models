import math
import torch
from torch import Tensor

from typing import Optional, Tuple, List


YoloNasPoseDecodedPredictions = Tuple[Tensor, Tensor, Tensor, Tensor]
YoloNasPoseRawOutputs = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[int], Tensor]


@torch.no_grad()
def generate_anchors_for_grid_cell(
    feats: Tuple[Tensor, ...],
    fpn_strides: Tuple[int, ...],
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, Tensor, List[int], Tensor]:
    """
    Like ATSS, generate anchors based on grid size.

    :param feats: shape[s, (b, c, h, w)]
    :param fpn_strides: shape[s], stride for each scale feature
    :param grid_cell_size: anchor size
    :param grid_cell_offset: The range is between 0 and 1.
    :param dtype: Type of the anchors.

    :return:
        - anchors: shape[l, 4], "xmin, ymin, xmax, ymax" format.
        - anchor_points: shape[l, 2], "x, y" format.
        - num_anchors_list: shape[s], contains [s_1, s_2, ...].
        - stride_tensor: shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    device = feats[0].device
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride

        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        anchor = torch.stack(
            [shift_x - cell_half_size, shift_y - cell_half_size, shift_x + cell_half_size, shift_y + cell_half_size],
            dim=-1,
        ).to(dtype=dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype))

    anchors = torch.cat(anchors).to(device)
    anchor_points = torch.cat(anchor_points).to(device)
    stride_tensor = torch.cat(stride_tensor).to(device)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def batch_distance2bbox(points: Tensor, distance: Tensor, max_shapes: Optional[Tensor] = None) -> Tensor:
    """Decode distance prediction to bounding box for batch.

    :param points: [B, ..., 2], "xy" format
    :param distance: [B, ..., 4], "ltrb" format
    :param max_shapes: [B, 2], "h,w" format, Shape of the image.
    :return: Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, dim=-1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = points - lt
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], dim=-1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def width_multiplier(original, factor, divisor: int = None):
    if divisor is None:
        return int(original * factor)
    else:
        return math.ceil(int(original * factor) / divisor) * divisor