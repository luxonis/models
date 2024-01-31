from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from luxonis_ml.loader import LabelType
from torch import Tensor

from luxonis_train.losses.luxonis_loss import LuxonisLoss

from luxonis_train.utils.types import Labels, ModulePacket


class BCEWithLogitsLoss(LuxonisLoss[Tensor, Tensor]):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        pos_weight: Optional[Tensor] = None,
        **kwargs,
    ):
        """Pytorch BCEWithLogitsLoss wrapper. For attribute definitions check
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        super().__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight if pos_weight is not None else None,
        )

    def validate(self, outputs: ModulePacket, labels: Labels) -> None:
        ...

    def preprocess(
        self, outputs: ModulePacket, labels: Labels
    ) -> tuple[Tensor, Tensor]:
        if "class" in outputs:
            return outputs["class"][0], labels[LabelType.CLASSIFICATION]
        elif "segmentation" in outputs:
            return outputs["segmentation"][0], labels[LabelType.SEGMENTATION]
        raise RuntimeError("No valid output found for BCEWithLogitsLoss")

    def compute_loss(
        self, preds: Tensor, target: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if preds.ndim != target.ndim:
            raise RuntimeError(
                f"Target tensor dimension ({target.ndim}) and preds tensor"
                f" dimension ({preds.ndim}) should be the same."
            )
        return self.criterion(preds, target), {}


class SmoothBCEWithLogitsLoss(LuxonisLoss[Tensor, Tensor]):
    def __init__(self, label_smoothing: float = 0.0, bce_pow: float = 1.0, **kwargs):
        """BCE with logits loss with label smoothing

        Args:
            label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0.
            bce_pow (float, optional): Weight for positive samples. Defaults to 1.0.
        """
        super().__init__(**kwargs)
        self.negative_smooth_const = 1.0 - 0.5 * label_smoothing
        self.positive_smooth_const = 0.5 * label_smoothing
        self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pow]))

    def validate(self, outputs: ModulePacket, labels: Labels) -> None:
        ...

    def preprocess(
        self, outputs: ModulePacket, labels: Labels
    ) -> tuple[Tensor, Tensor]:
        if "class" in outputs:
            return outputs["class"][0], labels[LabelType.CLASSIFICATION]
        elif "segmentation" in outputs:
            return outputs["segmentation"][0], labels[LabelType.SEGMENTATION]
        raise RuntimeError("No valid output found for BCEWithLogitsLoss")

    def compute_loss(
        self, predictions: list[Tensor], target: Tensor
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """
        Computes the BCE loss with label smoothing.

        Args:
            prediction (torch.Tensor): A tensor of shape (N, n_classes),
                containing the predicted class scores.
            target (torch.Tensor): A tensor of shape (N,), containing the
                ground-truth class labels

        Returns:
            torch.Tensor: A scalar tensor.
        """
        prediction = predictions[0]
        smoothed_target = torch.full_like(
            prediction,
            self.negative_smooth_const,
            device=prediction.device,
        )
        smoothed_target[torch.arange(len(target)), target] = self.positive_smooth_const
        return self.criterion(prediction, smoothed_target), {}
