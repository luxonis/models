import torch
import time
import math
from typing import List
from torchvision.ops import box_convert, nms


def dist2bbox(distance, anchor_points, box_format="xyxy"):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == "xyxy":
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == "xywh":
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist


def bbox_iou(
    bbox1: torch.Tensor,
    bbox2: torch.Tensor,
    box_format: str = "xywh",
    iou_type: str = "none",
    eps: float = 1e-7,
):
    """Caclulate iou between boxs

    Args:
        bbox1 (torch.Tensor): first bbox
        bbox2 (torch.Tensor): second bbox
        box_format (str, optional): Input box format, must be one of "xywh" or "xywh". Defaults to "xywh".
        iou_type (str, optional): Can be one of "none", "ciou", "diou, "giou" or "siou". Defaults to "none".
        eps (float, optional): Value to avoid divide by zero error. Defaults to 1e-7.
    """
    if bbox1.shape[0] != bbox2.shape[0]:
        bbox2 = bbox2.T
        if box_format == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        elif box_format == "xywh":
            b1_x1, b1_x2 = bbox1[0] - bbox1[2] / 2, bbox1[0] + bbox1[2] / 2
            b1_y1, b1_y2 = bbox1[1] - bbox1[3] / 2, bbox1[1] + bbox1[3] / 2
            b2_x1, b2_x2 = bbox2[0] - bbox2[2] / 2, bbox2[0] + bbox2[2] / 2
            b2_y1, b2_y2 = bbox2[1] - bbox2[3] / 2, bbox2[1] + bbox2[3] / 2
    else:
        if box_format == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(bbox1, 1, dim=-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(bbox2, 1, dim=-1)

        elif box_format == "xywh":
            b1_x1, b1_y1, b1_w, b1_h = torch.split(bbox1, 1, dim=-1)
            b2_x1, b2_y1, b2_w, b2_h = torch.split(bbox2, 1, dim=-1)
            b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
            b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
            b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
            b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    if iou_type == "giou":
        c_area = cw * ch + eps  # convex area
        iou = iou - (c_area - union) / c_area
    elif iou_type in ["diou", "ciou"]:
        c2 = cw**2 + ch**2 + eps  # convex diagonal squared
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
        ) / 4  # center distance squared
        if iou_type == "diou":
            iou = iou - rho2 / c2
        elif iou_type == "ciou":
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
            )
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            iou = iou - (rho2 / c2 + v * alpha)
    elif iou_type == "siou":
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
            1 - torch.exp(-1 * omiga_h), 4
        )
        iou = iou - 0.5 * (distance_cost + shape_cost)

    return iou


def non_max_suppression_bbox(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.
    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(
        prediction[..., 4] > conf_thres,
        torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres,
    )  # candidates
    # Check the parameters.
    assert (
        0 <= conf_thres <= 1
    ), f"conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided."
    assert (
        0 <= iou_thres <= 1
    ), f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = box_convert(x[:, :4], "cxcywh", "xyxy")

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (
                    box[box_idx],
                    x[box_idx, class_idx + 5, None],
                    class_idx[:, None].float(),
                ),
                1,
            )
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = (
            x[:, :4] + class_offset,
            x[:, 4],
        )  # boxes (offset by class), scores
        keep_box_idx = nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


# NOTE: might be able to merge this with non_max_suppression_kpts
def non_max_suppression_kpts(
    prediction,
    conf_thresh=0.03,
    iou_thresh=0.30,
    nc=1,
    max_det=300,
    max_wh=4096,
    time_limit=10.0,
    max_nms=30000,
    redundant=True,
    merge=False,
):
    """Runs NMS on keypoint predictions"""

    xc = prediction[..., 4] > conf_thresh  # candidates

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5 : 5 + nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = box_convert(x[:, :4], in_fmt="cxcywh", out_fmt="xyxy")

        kpts = x[:, 5 + nc :]
        conf, j = x[:, 5 : 5 + nc].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thresh]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thresh)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = (
                bbox_iou(boxes[i], boxes, box_format="xyxy") > iou_thresh
            )  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def anchors_from_dataset(
    loader: "torch.utils.data.DataLoader",
    n_anchors: int = 9,
    n_generations: int = 1000,
    ratio_threshold: float = 4.0,
):
    """Generates anchors based on bounding box annotations present in provided data loader.
    Adapted from: https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py

    Args:
        loader (torch.utils.data.DataLoader): Data loader
        n_anchors (int, optional): Number of anchors, this is normally num_heads * 3 which generates 3 anchors per layer. Defaults to 9.
        n_generations (int, optional): Number of iterations for anchor improvement with genetic algorithm. Defaults to 1000.
        ratio_threshold (float, optional): Minimum threshold for ratio. Defaults to 4.0.

    Returns:
        torch.Tensor: Proposed anchors
    """
    from scipy.cluster.vq import kmeans
    from luxonis_ml.loader import LabelType

    print("Generating anchors...")
    wh = []
    for batch in loader:
        inputs = batch[0]
        label_dict = batch[1]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        curr_wh = boxes[:, 4:]
        wh.append(curr_wh)
    _, _, h, w = inputs.shape  # assuming all images are same size
    img_size = torch.tensor([w, h])
    wh = torch.vstack(wh) * img_size

    # filter out small objects (w or h < 2 pixels)
    wh = wh[(wh >= 2).any(1)]

    # KMeans
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

    def calc_best_anchor_ratio(anchors, wh):
        """Calculate how well most suitable anchor box matches each target bbox"""
        symetric_size_ratios = torch.min(
            wh[:, None] / anchors[None], anchors[None] / wh[:, None]
        )
        worst_side_size_ratio = symetric_size_ratios.min(-1).values
        best_anchor_ratio = worst_side_size_ratio.max(-1).values
        return best_anchor_ratio

    def calc_best_possible_recall(anchors, wh):
        """Calculate best possible recall if every bbox is matched to an appropriate anchor"""
        best_anchor_ratio = calc_best_anchor_ratio(anchors, wh)
        best_possible_recall = (best_anchor_ratio > 1 / ratio_threshold).float().mean()
        return best_possible_recall

    def anchor_fitness(anchors, wh):
        """Fitness function used for anchor evolve"""
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
    print(
        f"Anchor generation finished. Best possible recall: {calc_best_possible_recall(proposed_anchors, wh)}"
    )

    return proposed_anchors


def anchors_for_fpn_features(
    features: List[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    is_eval: bool = False,
):
    """Generated anchor boxes, anchor points, number of anchors and
    strides based on FPN feature shapes and strides.

    Args:
        features (List[torch.Tensor]): List of FPN features
        strides (torch.Tensor): Strides of FPN features
        grid_cell_size (float, optional): Cell size in respect to input image size. Defaults to 5.0.
        grid_cell_offset (float, optional): Percent grid cell center's offset. Defaults to 0.5.
        is_eval (bool, optional): Weather return data is used for eval or not. Defaults to False.

    Returns:
        Tuple: Bbox anchors, center anchors, number of anchors, strides
    """
    anchors = []
    anchor_points = []
    n_anchors_list = []
    stride_tensor = []
    for feature, stride in zip(features, strides):
        _, _, h, w = feature.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = torch.arange(end=w) + grid_cell_offset
        shift_y = torch.arange(end=h) + grid_cell_offset
        if not is_eval:
            shift_x *= stride
            shift_y *= stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        # achor boxes
        anchor = (
            torch.stack(
                [
                    shift_x - cell_half_size,
                    shift_y - cell_half_size,
                    shift_x + cell_half_size,
                    shift_y + cell_half_size,
                ],
                axis=-1,
            )
            .reshape(-1, 4)
            .to(feature.dtype)
        )
        anchors.append(anchor)

        # achor box centers
        anchor_point = (
            torch.stack([shift_x, shift_y], axis=-1).reshape(-1, 2).to(feature.dtype)
        )
        anchor_points.append(anchor_point)

        n_anchors_list.append(h * w)
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=feature.dtype))

    device = feature.device
    anchors = torch.cat(anchors).to(device)
    anchor_points = torch.cat(anchor_points).to(device)
    stride_tensor = torch.cat(stride_tensor).to(device)
    return anchors, anchor_points, n_anchors_list, stride_tensor
