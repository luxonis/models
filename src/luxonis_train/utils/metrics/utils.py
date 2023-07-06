import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert

from .custom import ObjectKeypointSimilarity
from luxonis_train.utils.head_type import *
from luxonis_train.models.heads import *
from luxonis_train.utils.head_utils import yolov6_out2box
from luxonis_train.utils.boxutils import non_max_suppression_kpts

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
    if isinstance(head.type, Classification):
        collection = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes),
            "precision": torchmetrics.Precision(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes),
            "recall": torchmetrics.Recall(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes),
            "f1": torchmetrics.F1Score(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes)
        })
    elif isinstance(head.type, MultiLabelClassification):
        collection = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(task="multilabel", num_labels=head.n_classes),
            "precision": torchmetrics.Precision(task="multilabel", num_labels=head.n_classes),
            "recall": torchmetrics.Recall(task="multilabel", num_labels=head.n_classes),
            "f1": torchmetrics.F1Score(task="multilabel", num_labels=head.n_classes)
        })
    elif isinstance(head.type, SemanticSegmentation):
        collection = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes, ignore_index=0 if is_binary else None),
            "mIoU": torchmetrics.JaccardIndex(task="binary" if is_binary else "multiclass",
                num_classes=head.n_classes, ignore_index=0 if is_binary else None),

        })
    elif isinstance(head.type, ObjectDetection):
        collection = torchmetrics.MetricCollection({
            "mAP": MeanAveragePrecision(box_format="xyxy")
        })
    elif isinstance(head.type, KeyPointDetection):
        collection = torchmetrics.MetricCollection({
            "oks": ObjectKeypointSimilarity()
        })

    return nn.ModuleDict({
        "train_metrics": collection,
        "val_metrics": collection.clone(),
        "test_metrics": collection.clone(),
    })


def postprocess_for_metrics(output: torch.Tensor, labels: torch.Tensor, head: nn.Module):
    """ Performs post-processing on output and labels for specific metrics"""
    if isinstance(head.type, Classification):
        if head.n_classes != 1:
            labels = torch.argmax(labels, dim=1)
            output = torch.argmax(output, dim=1)
        return output, labels
    elif isinstance(head.type, MultiLabelClassification):
        return output, labels
    elif isinstance(head.type, SemanticSegmentation):
        if head.n_classes != 1:
            labels = torch.argmax(labels, dim=1)
        return output, labels
    elif isinstance(head.type, ObjectDetection):
        if isinstance(head, YoloV6Head):
            output, labels = yolov6_to_metrics(output, labels, head)
            return output, labels
    elif isinstance(head.type, KeyPointDetection):
        if isinstance(head, IKeypoint):
            output, labels = yolov7_pose_to_metrics(output[0], labels, head)
            return output, labels


def yolov6_to_metrics(output: torch.Tensor, labels: torch.Tensor, head: nn.Module):
    """ Performs post-processing on ouptut and labels for YoloV6 output"""
    kwargs = {"conf_thres":0.001, "iou_thres": 0.6}
    output_nms = yolov6_out2box(output, head, **kwargs)
    image_size = head.original_in_shape[2:]

    output_list = []
    labels_list = []
    for i in range(len(output_nms)):
        output_list.append({
            "boxes": output_nms[i][:,:4],
            "scores": output_nms[i][:,4],
            "labels": output_nms[i][:,5]
        })

        curr_labels = labels[labels[:,0]==i]
        curr_bboxs = box_convert(curr_labels[:, 2:], "xywh", "xyxy")
        curr_bboxs[:, 0::2] *= image_size[1]
        curr_bboxs[:, 1::2] *= image_size[0]
        labels_list.append({
            "boxes": curr_bboxs,
            "labels": curr_labels[:,1]
        })

    return output_list, labels_list

def yolov7_pose_to_metrics(output: torch.Tensor, labels: torch.Tensor, head: nn.Module):
    labels = labels.to(output.device)
    nms = non_max_suppression_kpts(output)
    output_list = []
    labels_list = []
    image_size = head.original_in_shape[2:]
    for i in range(len(nms)):
        output_list.append({
            "boxes": nms[i][:, :4],
            "scores": nms[i][:, 4],
            "labels": nms[i][:, 5],
        })

        curr_labels = labels[labels[:, 0] == i]
        curr_bboxs = box_convert(curr_labels[:, 2: 6], "cxcywh", "xyxy")
        curr_bboxs[:, 0::2] *= image_size[1]
        curr_bboxs[:, 1::2] *= image_size[0]
        labels_list.append({
            "boxes": curr_bboxs,
            "labels": curr_labels[:, 1],
        })
    return output_list, labels_list