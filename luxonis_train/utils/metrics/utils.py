import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from luxonis_train.utils.head_type import *
from luxonis_train.models.heads import *
from luxonis_train.utils.head_utils import yolov6_out2box
from luxonis_train.utils.boxutils import xywh2xyxy_coco

def init_metrics(head):
    if isinstance(head.type, Classification):
        collection = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=head.n_classes),
            "precision": torchmetrics.Precision(task="multiclass", num_classes=head.n_classes),
            "recall": torchmetrics.Recall(task="multiclass", num_classes=head.n_classes),
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=head.n_classes)
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
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=head.n_classes),
            "mIoU": torchmetrics.JaccardIndex(task="multiclass", num_classes=head.n_classes),

        })
    elif isinstance(head.type, ObjectDetection):
        collection = torchmetrics.MetricCollection({
            "mAP": MeanAveragePrecision(box_format="xyxy")
        })

    return nn.ModuleDict({
        "train_metrics": collection,
        "val_metrics": collection.clone(),
        "test_metrics": collection.clone(),
    })


def postprocess_for_metrics(output, labels, head):
    if isinstance(head.type, Classification):
        labels = torch.argmax(labels, dim=1)
        return output, labels
    elif isinstance(head.type, MultiLabelClassification):
        return output, labels
    elif isinstance(head.type, SemanticSegmentation):
        labels = torch.argmax(labels, dim=1)
        return output, labels
    elif isinstance(head.type, ObjectDetection):
        if isinstance(head, YoloV6Head):
            output, labels = yolov6_to_metrics(output, labels, head)
            return output, labels


def yolov6_to_metrics(output, labels, head):
    kwargs = {"conf_thres":0.001, "iou_thres": 0.6}
    output_nms = yolov6_out2box(output, head, **kwargs)
    img_shape = head.original_in_shape[2:]

    output_list = []
    labels_list = []
    for i in range(len(output_nms)):
        output_list.append({
            "boxes": output_nms[i][:,:4],
            "scores": output_nms[i][:,4],
            "labels": output_nms[i][:,5]
        })
        
        curr_labels = labels[labels[:,0]==i]
        curr_bboxs = xywh2xyxy_coco(curr_labels[:, 2:])
        curr_bboxs[:, 0::2] *= img_shape[1]
        curr_bboxs[:, 1::2] *= img_shape[0]
        labels_list.append({
            "boxes": curr_bboxs,
            "labels": curr_labels[:,1]
        })

    return output_list, labels_list