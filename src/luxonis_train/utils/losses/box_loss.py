from typing import Tuple

import torch
from torch import Tensor, nn

from luxonis_train.utils.boxutils import bbox_iou


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
        x_y = prediction[:, :2].sigmoid() * 2.0 - 0.5
        w_h = (prediction[:, 2:].sigmoid() * 2) ** 2 * anchor.to(device)
        boxes = torch.cat((x_y, w_h), 1).T
        iou = bbox_iou(boxes, target.to(device), box_format="xywh", iou_type="ciou")
        return (1.0 - iou).mean(), iou
