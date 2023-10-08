# from .custom import (
#     ObjectKeypointSimilarity,
#     MeanAveragePrecision,
#     MeanAveragePrecisionKeypoints,
# )
# from luxonis_train.utils.constants import HeadType


import torch.nn as nn
import torchmetrics
from torch import Tensor
from typing import List, Dict, Any, Tuple

from luxonis_train.utils.registry import METRICS
from luxonis_train.utils.general import flatten_dict

"""
Default average method for different metrics:
Accuract: micro
Precision: micro
Recall: micro
F1Score: micro
JaccardIndex: macro
"""


METRICS.register_module(module=torchmetrics.Accuracy)
METRICS.register_module(module=torchmetrics.Precision)
METRICS.register_module(module=torchmetrics.Recall)
METRICS.register_module(module=torchmetrics.F1Score)
METRICS.register_module(module=torchmetrics.JaccardIndex)
METRICS.register_module(module=torchmetrics.Dice)


class MetricModule:
    def __init__(self, cfg: List[Dict[str, Any]]):
        """Metric module with metrics for train, val and test view

        Args:
            cfg (List[Dict[str, Any]]): List of init parameters for the metrics
        """

        collection = self._build_collection(cfg)
        self.metrics = {
            "train": collection,
            "val": collection.clone(),
            "test": collection.clone(),
        }
        self.all_metrics = list(
            self.metrics["train"].keys()
        )  # list of all present metrics

    def update(self, view: str, metric_mapping: Dict[str, Tuple[Any, Any]]) -> None:
        """Calls update method on each metric from the view with correct preds and target data

        Args:
            view (str): Which MetricCollection to use
            metric_mapping (Dict[str, Tuple[Any, Any]]): Mapping between metric name
                and specific preds and target tuple
        """
        for metric_name in self.metrics[view]:
            if metric_name in metric_mapping:
                self.metrics[view][metric_name].update(*metric_mapping[metric_name])
            else:
                self.metrics[view][metric_name].update(*metric_mapping["raw"])

    def compute(self, view: str) -> Dict[str, Tensor | Dict[str, Tensor]]:
        """Calls compute method on each metric from the view and returns the final results

        Args:
            view (str): Which MetricCollection to use

        Returns:
            Dict[str, Tensor | Dict[str, Tensor]]: Mapping between metric name and final results
        """
        outputs = {}
        for metric_name in self.metrics[view]:
            outputs[metric_name] = self.metrics[view][metric_name].compute()
        outputs = flatten_dict(outputs)
        return outputs

    def reset(self, view: str) -> None:
        """Resets all metric from the view

        Args:
            view (str): Which MetricCollection to use
        """
        self.metrics[view].reset()

    def _build_collection(
        self, cfg: List[Dict[str, Any]]
    ) -> torchmetrics.MetricCollection:
        """Builds MetricCollection based on provided config

        Args:
            cfg (List[Dict[str, Any]]): List of init parameters for the metrics

        Returns:
            torchmetrics.MetricCollection: Ouptut MetricCollection object
        """
        all_metrics = []
        for metric in cfg:
            all_metrics.append(METRICS.get(metric["name"])(**metric.get("params", {})))
        collection = torchmetrics.MetricCollection(all_metrics)
        return collection


# def init_metrics(head: nn.Module):
#     """Initializes specific metrics depending on the head type and returns nn.ModuleDict"""

#     is_binary = head.n_classes == 1

#     metrics = {}
#     for head_type in head.head_types:
#         if head_type == HeadType.CLASSIFICATION:
#             metrics["accuracy"] = torchmetrics.Accuracy(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#             metrics["precision"] = torchmetrics.Precision(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#             metrics["recall"] = torchmetrics.Recall(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#             metrics["f1"] = torchmetrics.F1Score(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#         elif head_type == HeadType.MULTI_LABEL_CLASSIFICATION:
#             metrics["accuracy"] = torchmetrics.Accuracy(
#                 task="multilabel", num_labels=head.n_classes
#             )
#             metrics["precision"] = torchmetrics.Precision(
#                 task="multilabel", num_labels=head.n_classes
#             )
#             metrics["recall"] = torchmetrics.Recall(
#                 task="multilabel", num_labels=head.n_classes
#             )
#             metrics["f1"] = torchmetrics.F1Score(
#                 task="multilabel", num_labels=head.n_classes
#             )
#         elif head_type == HeadType.SEMANTIC_SEGMENTATION:
#             metrics["mIoU"] = torchmetrics.JaccardIndex(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#             metrics["accuracy"] = torchmetrics.Accuracy(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#             metrics["f1"] = torchmetrics.F1Score(
#                 task="binary" if is_binary else "multiclass", num_classes=head.n_classes
#             )
#         elif head_type == HeadType.OBJECT_DETECTION:
#             metrics["map"] = MeanAveragePrecision(
#                 box_format="xyxy", class_metrics=False if is_binary else True
#             )
#         elif head_type == HeadType.KEYPOINT_DETECTION:
#             metrics["oks"] = ObjectKeypointSimilarity(num_keypoints=head.n_keypoints)
#         else:
#             raise KeyError(
#                 f"No metrics for head type = {head_type} are currently supported."
#             )

#     # metrics for specific HeadType combinations
#     if all(
#         head_type in head.head_types
#         for head_type in [HeadType.OBJECT_DETECTION, HeadType.KEYPOINT_DETECTION]
#     ):
#         metrics["kpt_map"] = MeanAveragePrecisionKeypoints(
#             box_format="xyxy", num_keypoints=head.n_keypoints
#         )

#     collection = torchmetrics.MetricCollection(metrics)

#     return nn.ModuleDict(
#         {
#             "train_metrics": collection,
#             "val_metrics": collection.clone(),
#             "test_metrics": collection.clone(),
#         }
#     )
