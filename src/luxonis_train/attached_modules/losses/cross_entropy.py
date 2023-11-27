# TODO: Add docstring

from typing import Literal

import torch.nn as nn
from torch import Tensor

from .luxonis_loss import LuxonisLoss


class CrossEntropyLoss(LuxonisLoss[Tensor, Tensor]):
    """Pytorch CrossEntropyLoss wrapper.

    For attribute definitions check
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.ndim == target.ndim:
            ch_dim = 1 if preds.ndim > 1 else 0
            target = target.argmax(dim=ch_dim)

        # target.ndim must be preds.ndim-1 since target should contain class indices
        if target.ndim != preds.ndim - 1:
            raise RuntimeError(
                f"Target tensor dimension should equeal to preds dimension - 1 ({preds.ndim-1}) "
                f"but is ({target.ndim})."
            )
        return self.criterion(preds, target)
