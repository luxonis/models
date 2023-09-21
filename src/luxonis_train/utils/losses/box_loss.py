from typing import Tuple

import torch
from torch import Tensor, nn

from luxonis_train.utils.boxutils import bbox_iou, process_bbox_predictions


class BoxLoss(nn.Module):
    def forward(
        self, prediction: Tensor, anchor: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the box loss for a batch of predictions, anchors and targets.

        Args:
            prediction (Tensor): A tensor of shape (n_detections, 4) containing
                the predicted bounding box coordinates (x, y, w, h).
            anchor (Tensor): A tensor of shape (n_detections, 2) containing the anchor
                box coordinates (w, h).
            target (Tensor): A tensor of shape (n_detections, 4) containing the
                target bounding box coordinates (x, y, w, h).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the box loss and the IoU
            between the predicted and target boxes. The box loss is a tensor of
            shape (1,) and the IoU is a tensor of shape (n_detections,).
        """
        device = prediction.device
        x_y, w_h, tail = process_bbox_predictions(prediction, anchor.to(device))
        boxes1 = torch.cat((x_y, w_h, tail), 1).T
        iou = bbox_iou(boxes1, target.to(device), box_format="xywh", iou_type="ciou")
        return (1.0 - iou).mean(), iou
