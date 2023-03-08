import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from luxonis_train.utils.head_type import *
from luxonis_train.utils.assigners.anchor_generator import generate_anchors
from luxonis_train.utils.boxutils import dist2bbox, non_max_suppression, xywh2xyxy_coco

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
    head_name = head.__class__.__name__
    if isinstance(head.type, Classification):
        labels = torch.argmax(labels, dim=1)
        return output, labels
    elif isinstance(head.type, MultiLabelClassification):
        return output, labels
    elif isinstance(head.type, SemanticSegmentation):
        labels = torch.argmax(labels, dim=1)
        return output, labels
    elif isinstance(head.type, ObjectDetection):
        if head_name == "YoloV6Head":
            output, labels = postprocess_yolov6(output, labels, head)
            return output, labels


def postprocess_yolov6(output, labels, head, **kwargs): 
    x, cls_score_list, reg_dist_list = output
    anchor_points, stride_tensor = generate_anchors(x, head.stride, 
        head.grid_cell_size, head.grid_cell_offset, is_eval=True)
    pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")

    pred_bboxes *= stride_tensor
    output_merged = torch.cat([
        pred_bboxes, 
        torch.ones((x[-1].shape[0], pred_bboxes.shape[1], 1), dtype=pred_bboxes.dtype, device=pred_bboxes.device), 
        cls_score_list 
    ], axis=-1)

    conf_thres = kwargs.get("conf_thres", 0.001)
    iou_thres = kwargs.get("iou_thres", 0.6)

    # output_nms = non_max_suppression(output_merged, conf_thres=conf_thres, iou_thres=iou_thres)
    output_nms = [torch.zeros((300,6)) for _ in range(len(output_merged))]
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