import pytest
import torch
from luxonis_train.utils.boxutils import (
    anchors_for_fpn_features,
    bbox2dist,
    bbox_iou,
    compute_iou_loss,
    dist2bbox,
    non_max_suppression,
    process_bbox_predictions,
    process_keypoints_predictions,
)


def generate_random_bboxes(num_bboxes, max_width, max_height, format="xyxy"):
    # Generate top-left corners (x1, y1)
    x1y1 = torch.rand(num_bboxes, 2) * torch.tensor([max_width - 1, max_height - 1])

    # Generate widths and heights ensuring x2 > x1 and y2 > y1
    wh = (
        torch.rand(num_bboxes, 2) * (torch.tensor([max_width, max_height]) - 1 - x1y1)
        + 1
    )

    if format == "xyxy":
        # Calculate bottom-right corners (x2, y2) for xyxy format
        x2y2 = x1y1 + wh
        bboxes = torch.cat((x1y1, x2y2), dim=1)
    elif format == "xywh":
        # Use x1y1 as top-left corner and wh as width and height for xywh format
        bboxes = torch.cat((x1y1, wh), dim=1)
    elif format == "cxcywh":
        # Calculate center coordinates and use wh as width and height for cxcywh format
        cxcy = x1y1 + wh / 2
        bboxes = torch.cat((cxcy, wh), dim=1)
    else:
        raise ValueError("Unsupported format. Choose from 'xyxy', 'xywh', 'cxcywh'.")

    return bboxes


def test_dist2bbox():
    distance = torch.rand(10, 4)
    anchor_points = torch.rand(10, 2)
    bbox = dist2bbox(distance, anchor_points)

    assert bbox.shape == distance.shape


def test_bbox2dist():
    bbox = torch.rand(10, 4)
    anchor_points = torch.rand(10, 2)
    reg_max = 10.0

    distance = bbox2dist(bbox, anchor_points, reg_max)

    assert distance.shape == bbox.shape


def test_bbox_iou():
    for format in ["xyxy", "cxcywh", "xywh"]:
        bbox1 = generate_random_bboxes(5, 640, 640, format)
        bbox2 = generate_random_bboxes(8, 640, 640, format)

        iou = bbox_iou(bbox1, bbox2)

        assert iou.shape == (5, 8)
        assert iou.min() >= 0 and iou.max() <= 1


def test_compute_iou_loss():
    pred_bboxes = generate_random_bboxes(8, 640, 640, "xyxy")
    target_bboxes = generate_random_bboxes(8, 640, 640, "xyxy")

    loss_iou, iou = compute_iou_loss(pred_bboxes, target_bboxes, iou_type="giou")

    assert isinstance(loss_iou, torch.Tensor)
    assert isinstance(iou, torch.Tensor)
    assert 0 <= iou.min() and iou.max() <= 1


def test_process_bbox_predictions():
    bbox = generate_random_bboxes(10, 64, 64, "xywh")
    data = torch.rand(10, 4)
    prediction = torch.concat([bbox, data], dim=-1)
    anchor = torch.rand(10, 2)

    out_bbox_xy, out_bbox_wh, out_bbox_tail = process_bbox_predictions(
        prediction, anchor
    )

    assert out_bbox_xy.shape == (10, 2)
    assert out_bbox_wh.shape == (10, 2)
    assert out_bbox_tail.shape == (10, 4)


def test_process_keypoints_predictions():
    keypoints = torch.rand(10, 15)  # 5 keypoints * 3 (x, y, visibility)

    x, y, visibility = process_keypoints_predictions(keypoints)

    assert x.shape == y.shape == visibility.shape == (10, 5)


def test_anchors_for_fpn_features():
    features = [torch.rand(1, 256, 14, 14), torch.rand(1, 256, 28, 28)]
    strides = torch.tensor([8, 16])

    anchors, anchor_points, n_anchors_list, stride_tensor = anchors_for_fpn_features(
        features, strides
    )

    assert isinstance(anchors, torch.Tensor)
    assert isinstance(anchor_points, torch.Tensor)
    assert isinstance(n_anchors_list, list)
    assert isinstance(stride_tensor, torch.Tensor)
    assert len(n_anchors_list) == len(features)


def test_non_max_suppression():
    # Initialize test data
    n_classes = 5
    bs = 4
    preds = torch.rand(bs, 100, 6)  # Simulate predictions with shape [batch_size, N, M]
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 10

    # Test valid input
    result = non_max_suppression(
        preds, n_classes, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
    )
    assert isinstance(result, list)
    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    assert all(tensor.shape[1] == preds.shape[-1] for tensor in result)
    assert all(tensor.size(0) <= max_det for tensor in result)

    # Test invalid confidence threshold
    with pytest.raises(ValueError):
        non_max_suppression(preds, n_classes, conf_thres=1.5)

    # Test invalid IoU threshold
    with pytest.raises(ValueError):
        non_max_suppression(preds, n_classes, iou_thres=-0.1)

    # Test bbox format conversion
    for bbox_format in ["xyxy", "xywh", "cxcywh"]:
        result_format = non_max_suppression(preds, n_classes, bbox_format=bbox_format)  # type: ignore
        assert isinstance(result_format, list)

    # Test agnostic mode
    result_agnostic = non_max_suppression(preds, n_classes, agnostic=True)
    assert isinstance(result_agnostic, list)

    # Test non-agnostic mode
    result_non_agnostic = non_max_suppression(preds, n_classes, agnostic=False)
    assert isinstance(result_non_agnostic, list)

    # Test real scenario
    n_classes = 1
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 300

    # Define bounding boxes and scores (x1, y1, x2, y2, score, class)
    # Two boxes overlap, one with higher confidence should be kept
    box1 = [0, 0, 50, 50, 0.9, 0]  # High confidence box
    box2 = [10, 10, 60, 60, 0.6, 0]  # Overlapping box with lower confidence
    box3 = [100, 100, 150, 150, 0.8, 0]  # Non-overlapping box

    # Simulate predictions tensor
    preds = torch.tensor([[box1, box2, box3]])

    # Run non_max_suppression
    result = non_max_suppression(
        preds, n_classes, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
    )

    # The expected result should only include box1 and box3
    assert len(result[0]) == 2
    assert torch.allclose(result[0][0, :4], torch.tensor(box1[:4], dtype=torch.float32))
    assert torch.allclose(result[0][1, :4], torch.tensor(box3[:4], dtype=torch.float32))
