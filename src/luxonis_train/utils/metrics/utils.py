import torch.nn as nn
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .custom import ObjectKeypointSimilarity
from luxonis_train.utils.constants import HeadType

"""
Default average method for different metrics:
Accuract: micro
Precision: micro
Recall: micro
F1Score: micro
JaccardIndex: macro
"""

def init_metrics(head: nn.Module):
    """ Initializes specific metrics depending on the head type and returns nn.ModuleDict """

    is_binary = head.n_classes == 1
    
    metrics = {}
    for head_type in head.head_types:
        if head_type == HeadType.CLASSIFICATION:
            metrics["accuracy"] = torchmetrics.Accuracy(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes)
            metrics["precision"] = torchmetrics.Precision(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes)
            metrics["recall"] = torchmetrics.Recall(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes)
            metrics["f1"] = torchmetrics.F1Score(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes)
        elif head_type == HeadType.MULTI_LABEL_CLASSIFICATION:
            metrics["accuracy"] =  torchmetrics.Accuracy(task="multilabel", num_labels=head.n_classes)
            metrics["precision"] =  torchmetrics.Precision(task="multilabel", num_labels=head.n_classes)
            metrics["recall"] =  torchmetrics.Recall(task="multilabel", num_labels=head.n_classes)
            metrics["f1"] =  torchmetrics.F1Score(task="multilabel", num_labels=head.n_classes)
        elif head_type == HeadType.SEMANTIC_SEGMENTATION:
            metrics["accuracy"] = torchmetrics.Accuracy(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes, ignore_index=0 if is_binary else None)
            metrics["mIoU"] = torchmetrics.JaccardIndex(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes, ignore_index=0 if is_binary else None)
        elif head_type == HeadType.OBJECT_DETECTION:
            metrics["map"] = MeanAveragePrecision(box_format="xyxy")
        elif head_type == HeadType.KEYPOINT_DETECTION:
            metrics["oks"] = ObjectKeypointSimilarity()
        else:
            raise KeyError(f"No metrics for head type = {head_type} are currently supported.")
    
    collection = torchmetrics.MetricCollection(metrics)
    
    return nn.ModuleDict({
        "train_metrics": collection,
        "val_metrics": collection.clone(),
        "test_metrics": collection.clone(),
    })