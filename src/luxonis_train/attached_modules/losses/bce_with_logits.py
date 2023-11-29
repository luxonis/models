# TODO: document

from typing import Literal

import torch.nn as nn
from torch import Tensor

from .base_loss import BaseLoss


class BCEWithLogitsLoss(BaseLoss[Tensor, Tensor]):
    """Pytorch BCEWithLogitsLoss wrapper.

    For attribute definitions check
    https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        pos_weight: Tensor | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight if pos_weight is not None else None,
        )

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        if predictions.ndim != target.ndim:
            raise RuntimeError(
                f"Target tensor dimension ({target.ndim}) and preds tensor "
                f"dimension ({predictions.ndim}) should be the same."
            )
        return self.criterion(predictions, target)
