import torchmetrics.detection as detection
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.utils.types import (
    BBoxProtocol,
    Labels,
    LabelType,
    Packet,
)

from .base_metric import BaseMetric


class MeanAveragePrecision(BaseMetric, detection.MeanAveragePrecision):
    r"""Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for
    object detection predictions.

    .. math::
        \text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i

    where :math:`AP_i` is the average precision for class :math:`i` and :math:`n` is the number of classes. The average
    precision is defined as the area under the precision-recall curve. For object detection the recall and precision are
    defined based on the intersection of union (IoU) between the predicted bounding boxes and the ground truth bounding
    boxes e.g. if two boxes have an IoU > t (with t being some threshold) they are considered a match and therefore
    considered a true positive. The precision is then defined as the number of true positives divided by the number of
    all detected boxes and the recall is defined as the number of true positives divided by the number of all ground
    boxes.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes``
          detection boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates, but can be changed
          using the ``box_format`` parameter. Only required when `iou_type="bbox"`.
        - ``scores`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing detection scores for the
          boxes.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed detection
          classes for the boxes.
        - ``masks`` (:class:`~torch.Tensor`): boolean tensor of shape ``(num_boxes, image_height, image_width)``
          containing boolean masks. Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground
          truth boxes of the format specified in the constructor. only required when `iou_type="bbox"`.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.
        - ``masks`` (:class:`~torch.Tensor`): boolean tensor of shape ``(num_boxes, image_height, image_width)``
          containing boolean masks. Only required when `iou_type="segm"`.
        - ``iscrowd`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0/1 values indicating
          whether the bounding box/masks indicate a crowd of objects. Value is optional, and if not provided it will
          automatically be set to 0.
        - ``area`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing the area of the object.
          Value is optional, and if not provided will be automatically calculated based on the bounding box/masks
          provided. Only affects which samples contribute to the `map_small`, `map_medium`, `map_large` values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~torch.Tensor`), global mean average precision
        - map_small: (:class:`~torch.Tensor`), mean average precision for small objects
        - map_medium:(:class:`~torch.Tensor`), mean average precision for medium objects
        - map_large: (:class:`~torch.Tensor`), mean average precision for large objects
        - mar_1: (:class:`~torch.Tensor`), mean average recall for 1 detection per image
        - mar_10: (:class:`~torch.Tensor`), mean average recall for 10 detections per image
        - mar_100: (:class:`~torch.Tensor`), mean average recall for 100 detections per image
        - mar_small: (:class:`~torch.Tensor`), mean average recall for small objects
        - mar_medium: (:class:`~torch.Tensor`), mean average recall for medium objects
        - mar_large: (:class:`~torch.Tensor`), mean average recall for large objects
        - map_50: (:class:`~torch.Tensor`) (-1 if 0.5 not in the list of iou thresholds), mean average precision at
          IoU=0.50
        - map_75: (:class:`~torch.Tensor`) (-1 if 0.75 not in the list of iou thresholds), mean average precision at
          IoU=0.75
        - map_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average precision per
          observed class
        - mar_100_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average recall for 100
          detections per image per observed class
        - classes (:class:`~torch.Tensor`), list of all observed classes
    """

    def __init__(self, **kwargs):
        super().__init__(
            protocol=BBoxProtocol,
            required_labels=[LabelType.BOUNDINGBOX],
            **kwargs,
        )
        self.metric = detection.MeanAveragePrecision()

    def update(self, outputs: list[dict[str, Tensor]], labels: list[dict[str, Tensor]]):
        self.metric.update(outputs, labels)

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        label = labels[LabelType.BOUNDINGBOX]
        output_nms = outputs["boxes"]

        image_size = self.node_attributes.original_in_shape[2:]

        output_list: list[dict[str, Tensor]] = []
        label_list: list[dict[str, Tensor]] = []
        for i in range(len(output_nms)):
            output_list.append(
                {
                    "boxes": output_nms[i][:, :4],
                    "scores": output_nms[i][:, 4],
                    "labels": output_nms[i][:, 5].int(),
                }
            )

            curr_label = label[label[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list.append({"boxes": curr_bboxs, "labels": curr_label[:, 1].int()})

        return output_list, label_list

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metric_dict = self.metric.compute()

        del metric_dict["classes"]
        del metric_dict["map_per_class"]
        del metric_dict["mar_100_per_class"]
        map = metric_dict.pop("map")

        return map, metric_dict
