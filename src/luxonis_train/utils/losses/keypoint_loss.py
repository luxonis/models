from typing import Tuple

import torch
from torch import Tensor, nn

from luxonis_train.utils.boxutils import process_keypoints_predictions

from .common import BCEWithLogitsLoss


class KeypointLoss(nn.Module):
    def __init__(self, bce_power: float = 1.0, **_):
        super().__init__()
        self.BCE = BCEWithLogitsLoss(pos_weight=torch.tensor([bce_power]))

    def forward(self, prediction: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the keypoint loss and visibility loss
        for a given prediction and target.

        Args:
            prediction (Tensor): The predicted tensor of shape
                (n_detections, n_keypoints * 3).
            target (Tensor): The target tensor of shape (n_detections, n_keypoints * 2).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the keypoint loss
            tensor of shape (1,) and the visibility loss tensor of shape (1,).
        """
        x, y, visibility_score = process_keypoints_predictions(prediction)

        mask = target[:, 0::2] != 0
        visibility_loss = self.BCE(visibility_score, mask.float())
        distance = (x - target[:, 0::2]) ** 2 + (y - target[:, 1::2]) ** 2

        loss_factor = (torch.sum(mask != 0) + torch.sum(mask == 0)) / (
            torch.sum(mask != 0) + 1e-9
        )
        loss = loss_factor * (torch.log(distance + 1 + 1e-9) * mask).mean()
        return loss, visibility_loss
