import torch
import contextlib
import io
from torch import Tensor
from typing import List, Optional, Dict, Any, Tuple, Literal
from torchmetrics import Metric, detection
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ObjectKeypointSimilarity(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_keypoints: List[Tensor]
    groundtruth_keypoints: List[Tensor]
    groundtruth_scales: List[Tensor]

    def __init__(
        self,
        num_keypoints: int,
        kpt_sigmas: Optional[Tensor] = None,
        use_cocoeval_oks: bool = False,
        **kwargs,
    ) -> None:
        """Object Keypoint Similarity metric for evaluating keypoint predictions

        Args:
            num_keypoints (int): Number of keypoints
            kpt_sigmas (Optional[Tensor], optional): Sigma for each keypoint to weigh its importance,
                if None use same weights for all. Defaults to None.
            use_cocoeval_oks (bool, optional): Whether to use same OKS formula as in COCOeval or use
                the one from definition. Defaults to False.

        As input to ``forward`` and ``update`` the metric accepts the following input:
        - preds (List[Dict[str, Tensor]]): A list consisting of dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.

        - `target` (List[Dict[str, Tensor]]): A list consisting of dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.
            - scales (torch.FloatTensor): Tensor of shape (N) where each value corresponds to scale of the bounding box.
                Scale of one bounding box is defined as sqrt(width*height) where width and height are unnormalized.

        """
        super().__init__(**kwargs)

        self.num_keypoints = num_keypoints
        if kpt_sigmas is not None and len(kpt_sigmas) != num_keypoints:
            raise ValueError(f"Expected kpt_sigmas to be of shape (num_keypoints).")
        self.kpt_sigmas = kpt_sigmas or torch.ones(num_keypoints)
        self.use_cocoeval_oks = use_cocoeval_oks

        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_scales", default=[], dist_reduce_fx=None)

    def update(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ) -> None:
        """Torchmetric update function"""
        for item in preds:
            keypoints = fix_empty_tensors(item["keypoints"])
            self.pred_keypoints.append(keypoints)

        for item in target:
            keypoints = fix_empty_tensors(item["keypoints"])
            self.groundtruth_keypoints.append(keypoints)
            self.groundtruth_scales.append(item["scales"])

    def compute(self) -> Tensor:
        """Torchmetric compute function"""
        self.kpt_sigmas = self.kpt_sigmas.to(
            self.device
        )  # explicitly move to current device
        image_mean_oks = torch.zeros(len(self.groundtruth_keypoints))
        # iterate over all images
        for i, (pred_kpts, gt_kpts, gt_scales) in enumerate(
            zip(
                self.pred_keypoints, self.groundtruth_keypoints, self.groundtruth_scales
            )
        ):
            gt_kpts = torch.reshape(gt_kpts, (-1, self.num_keypoints, 3))  # [N, K, 3]
            pred_kpts = torch.reshape(
                pred_kpts, (-1, self.num_keypoints, 3)
            )  # [M, K, 3]

            # compute OKS
            image_ious = self._compute_oks(pred_kpts, gt_kpts, gt_scales)  # [M, N]

            # perform linear sum assignment
            gt_indices, pred_indices = linear_sum_assignment(
                image_ious.cpu().numpy(), maximize=True
            )
            matched_ious = [image_ious[n, m] for n, m in zip(gt_indices, pred_indices)]
            # take mean as OKS for image
            image_mean_oks[i] = torch.tensor(matched_ious).mean()

        # final prediction is mean of OKS over all images
        final_oks = image_mean_oks.nanmean()

        return final_oks

    def _compute_oks(self, pred: Tensor, gt: Tensor, scales: Tensor) -> Tensor:
        """Compute Object Keypoint Similarity between every GT and prediction

        Args:
            pred (Tensor): Prediction tensor with shape [N, K, 3]
            gt (Tensor): GT tensor with shape [M, K, 3]
            scale (float): Scale tensor for every GT [M,]

        Returns:
            Tensor: Object Keypoint Similarity every pred and gt [M, N]
        """
        eps = 1e-7
        distances = (gt[:, None, :, 0] - pred[..., 0]) ** 2 + (
            gt[:, None, :, 1] - pred[..., 1]
        ) ** 2  # (N, M, 17)
        kpt_mask = gt[..., 2] != 0  # only compute on visible keypoints
        if self.use_cocoeval_oks:
            # use same formula as in COCOEval script here:
            # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229
            oks = (
                distances
                / (2 * self.kpt_sigmas) ** 2
                / (scales[:, None, None] + eps)
                / 2
            )
        else:
            # use same formula as defined here: https://cocodataset.org/#keypoints-eval
            oks = distances / ((scales[:, None, None] + eps) * self.kpt_sigmas) ** 2 / 2

        return (torch.exp(-oks) * kpt_mask[:, None]).sum(-1) / (
            kpt_mask.sum(-1)[:, None] + eps
        )


class MeanAveragePrecision(detection.MeanAveragePrecision):
    def __init__(self, **kwargs):
        """Wrapper for torchmetrics.detection.MeanAveragePrecision that omits some of the returned values
        Check original documentation: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
        """
        super().__init__(**kwargs)

    def compute(self) -> dict:
        metric_dict = super().compute()
        # per class metrics are omitted until we find a nice way to log them
        del metric_dict["classes"]
        del metric_dict["map_per_class"]
        del metric_dict["mar_100_per_class"]

        return metric_dict


class MeanAveragePrecisionKeypoints(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_boxes: List[Tensor]
    pred_scores: List[Tensor]
    pred_labels: List[Tensor]
    pred_keypoints: List[Tensor]

    groundtruth_boxes: List[Tensor]
    groundtruth_labels: List[Tensor]
    groundtruth_area: List[Tensor]
    groundtruth_crowds: List[Tensor]
    groundtruth_keypoints: List[Tensor]

    def __init__(
        self,
        num_keypoints: int,
        kpt_sigmas: Optional[Tensor] = None,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        **kwargs,
    ):
        """Mean Average Precision metrics that uses OKS as IoU measure (adapted from: https://github.com/Lightning-AI/torchmetrics/blob/v1.0.1/src/torchmetrics/detection/mean_ap.py#L543)

        Args:
            num_keypoints (int): Number of keypoints
            kpt_sigmas (Optional[Tensor], optional): Sigma for each keypoint to weigh its importance,
                if None use same weights for all. Defaults to None.
            box_format (Literal[xyxy, xywh, cxcywh], optional): Input bbox format. Defaults to "xyxy".

        As input to ``forward`` and ``update`` the metric accepts the following input:
        - preds (List[Dict[str, Tensor]]): A list consisting of dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:
            - boxes (torch.FloatTensor): Tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` detection
                boxes of the format specified in the constructor. By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
            - scores (torch.FloatTensor): Tensor of shape ``(num_boxes)`` containing detection scores for the boxes.
            - labels (torch.IntTensor): Tensor of shape ``(num_boxes)`` containing 0-indexed detection classes for the boxes.
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.

        - `target` (List[Dict[str, Tensor]]): A list consisting of dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:
            - boxes (torch.FloatTensor): Tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground truth
                boxes of the format specified in the constructor. By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
            - labels: :class:`~torch.IntTensor` of shape ``(num_boxes)`` containing 0-indexed ground truth classes for the boxes.
            - iscrow (torch.IntTensor): Tensor of shape ``(num_boxes)`` containing 0/1 values indicating whether
                the bounding box/masks indicate a crowd of objects. Value is optional, and if not provided it will
                automatically be set to 0.
            - area (torch.FloatTensor): Tensor of shape ``(num_boxes)`` containing the area of the object. Value if
                optional, and if not provided will be automatically calculated based on the bounding box/masks provided.
                Only affects which samples contribute to the `map_small`, `map_medium`, `map_large` values
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.
        """
        super().__init__(**kwargs)

        self.num_keypoints = num_keypoints
        if kpt_sigmas is not None and len(kpt_sigmas) != num_keypoints:
            raise ValueError(f"Expected kpt_sigmas to be of shape (num_keypoints).")
        self.kpt_sigmas = kpt_sigmas or torch.ones(num_keypoints)

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format

        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)

        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_area", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_crowds", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)

    def update(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ) -> None:
        """Torchmetric update function"""
        for item in preds:
            boxes, keypoints = self._get_safe_item_values(item)
            self.pred_boxes.append(boxes)
            self.pred_keypoints.append(keypoints)
            self.pred_scores.append(item["scores"])
            self.pred_labels.append(item["labels"])

        for item in target:
            boxes, keypoints = self._get_safe_item_values(item)
            self.groundtruth_boxes.append(boxes)
            self.groundtruth_keypoints.append(keypoints)
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_area.append(
                item.get("area", torch.zeros_like(item["labels"]))
            )
            self.groundtruth_crowds.append(
                item.get("iscrowd", torch.zeros_like(item["labels"]))
            )

    def compute(self) -> Dict[str, Tensor]:
        """Torchmetric compute function"""
        coco_target, coco_preds = COCO(), COCO()
        coco_target.dataset = self._get_coco_format(
            self.groundtruth_boxes,
            self.groundtruth_keypoints,
            self.groundtruth_labels,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
        )
        coco_preds.dataset = self._get_coco_format(
            self.pred_boxes,
            self.pred_keypoints,
            self.groundtruth_labels,
            scores=self.pred_scores,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

            self.coco_eval = COCOeval(coco_target, coco_preds, iouType="keypoints")
            self.coco_eval.params.kpt_oks_sigmas = self.kpt_sigmas.cpu().numpy()

            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
            stats = self.coco_eval.stats

        return {
            "kpt_map": torch.tensor([stats[0]], dtype=torch.float32),
            "kpt_map_50": torch.tensor([stats[1]], dtype=torch.float32),
            "kpt_map_75": torch.tensor([stats[2]], dtype=torch.float32),
            "kpt_map_medium": torch.tensor([stats[3]], dtype=torch.float32),
            "kpt_map_large": torch.tensor([stats[4]], dtype=torch.float32),
            "kpt_mar": torch.tensor([stats[5]], dtype=torch.float32),
            "kpt_mar_50": torch.tensor([stats[6]], dtype=torch.float32),
            "kpt_mar_75": torch.tensor([stats[7]], dtype=torch.float32),
            "kpt_mar_medium": torch.tensor([stats[8]], dtype=torch.float32),
            "kpt_mar_large": torch.tensor([stats[9]], dtype=torch.float32),
        }

    def _get_coco_format(
        self,
        boxes: List[Tensor],
        keypoints: List[Tensor],
        labels: List[Tensor],
        scores: Optional[List[Tensor]] = None,
        crowds: Optional[List[Tensor]] = None,
        area: Optional[List[Tensor]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Transforms and returns all cached targets or predictions in COCO format.
        Format is defined at https://cocodataset.org/#format-data
        """
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, (image_boxes, image_kpts, image_labels) in enumerate(
            zip(boxes, keypoints, labels)
        ):
            image_boxes = image_boxes.cpu().tolist()
            image_kpts = image_kpts.cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({"id": image_id})

            for k, (image_box, image_kpt, image_label) in enumerate(
                zip(image_boxes, image_kpts, image_labels)
            ):
                if len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if len(image_kpt) != 3 * self.num_keypoints:
                    raise ValueError(
                        f"Invalid input keypoints of sample {image_id}, element {k} (expected 3*{self.num_keypoints} values, got {len(image_kpt)})"
                    )

                if type(image_label) != int:
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                if area is not None and area[image_id][k].cpu().tolist() > 0:
                    area_stat = area[image_id][k].cpu().tolist()
                else:
                    area_stat = image_box[2] * image_box[3]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": image_box,
                    "area": area_stat,
                    "category_id": image_label,
                    "iscrowd": crowds[image_id][k].cpu().tolist()
                    if crowds is not None
                    else 0,
                    "keypoints": image_kpt,
                    "num_keypoints": self.num_keypoints,
                }

                if scores is not None:
                    score = scores[image_id][k].cpu().tolist()
                    if type(score) != float:
                        raise ValueError(
                            f"Invalid input score of sample {image_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {"images": images, "annotations": annotations, "categories": classes}

    def _get_safe_item_values(self, item: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Convert and return the boxes"""
        boxes = fix_empty_tensors(item["boxes"])
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xywh")
        keypoints = fix_empty_tensors(item["keypoints"])
        return boxes, keypoints

    def _get_classes(self) -> List[int]:
        """Return a list of unique classes found in ground truth and detection data."""
        if len(self.pred_labels) > 0 or len(self.groundtruth_labels) > 0:
            return (
                torch.cat(self.pred_labels + self.groundtruth_labels)
                .unique()
                .cpu()
                .tolist()
            )
        return []


def fix_empty_tensors(input_tensor: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if input_tensor.numel() == 0 and input_tensor.ndim == 1:
        return input_tensor.unsqueeze(0)
    return input_tensor
