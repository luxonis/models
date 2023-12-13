# TODO: document

from typing import Literal

import torch
from torch import Tensor

from luxonis_train.attached_modules.losses import BaseLoss

from .cross_entropy import CrossEntropyLoss


class SoftmaxFocalLoss(BaseLoss[Tensor, Tensor]):
    def __init__(
        self,
        alpha: float | list[float] = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Focal loss implementation for multi-class/multi-label tasks using Softmax.

        Args:
            alpha (float | list[float], optional): Either a float for all channels or
            list of alphas for each channel with length C. Defaults to 0.25.
            gamma (float, optional): Defaults to 2.0.
            reduction (Literal["none", "mean", "sum"], optional): Defaults to "mean".
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_criterion = CrossEntropyLoss(reduction="none", **kwargs)

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.ce_criterion.forward(predictions, target)
        pt = torch.exp(-ce_loss)
        loss = ce_loss * ((1 - pt) ** self.gamma)

        if isinstance(self.alpha, float) and self.alpha >= 0:
            loss = self.alpha * loss
        elif isinstance(self.alpha, list):
            alpha_t = torch.tensor(self.alpha)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
