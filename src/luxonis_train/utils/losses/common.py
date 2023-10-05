import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Literal, Tuple, Union, List
from torchvision.ops import sigmoid_focal_loss

from luxonis_train.utils.losses.base_loss import BaseLoss
from luxonis_train.utils.registry import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(BaseLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        """Pytorch CrossEntropyLoss wrapper. For attribute definitions check
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        """
        super().__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(
        self, preds: Tensor, target: Tensor, epoch: int, step: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if preds.ndim == target.ndim:
            # argmax along channels dimension
            ch_dim = 1 if preds.ndim > 1 else 0
            target = target.argmax(dim=ch_dim)
        # target.ndim must be preds.ndim-1 since target should contain class indices
        if target.ndim != (preds.ndim - 1):
            raise RuntimeError(
                f"Target tensor dimension should equeal to preds dimension - 1 ({preds.ndim-1}) "
                f"but is ({target.ndim})."
            )
        return self.criterion(preds, target), {}


@LOSSES.register_module()
class BCEWithLogitsLoss(BaseLoss):
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
            pos_weight=pos_weight,
        )

    def forward(
        self, preds: Tensor, target: Tensor, epoch: int, step: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if preds.ndim != target.ndim:
            raise RuntimeError(
                f"Target tensor dimension ({target.ndim}) and preds tensor"
                f" dimension ({preds.ndim}) should be the same."
            )
        return self.criterion(preds, target), {}


@LOSSES.register_module()
class SmoothBCEWithLogitsLoss(BaseLoss):
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

    def forward(
        self, prediction: Tensor, target: Tensor, epoch: int, step: int
    ) -> Tensor:
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
        smoothed_target = torch.full_like(
            prediction,
            self.negative_smooth_const,
            device=prediction.device,
        )
        smoothed_target[torch.arange(len(target)), target] = self.positive_smooth_const
        return self.criterion(prediction, smoothed_target), {}


@LOSSES.register_module()
class SigmoidFocalLoss(BaseLoss):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Focal loss from `Focal Loss for Dense Object Detection`,
        https://arxiv.org/abs/1708.02002

        Args:
            alpha (float, optional): Defaults to 0.8.
            gamma (float, optional): Defaults to 2.0.
            reduction (Literal["none", "mean", "sum"], optional): Defaults to "mean".
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, preds: Tensor, target: Tensor, epoch: int, step: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if preds.ndim != target.ndim:
            raise RuntimeError(
                f"Target tensor dimension ({target.ndim}) and preds tensor"
                f" dimension ({preds.ndim}) should be the same."
            )
        loss = sigmoid_focal_loss(
            preds, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )

        return loss, {}


@LOSSES.register_module()
class SoftmaxFocalLoss(BaseLoss):
    def __init__(
        self,
        alpha: Union[float, List[float]] = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Focal loss implementation for multi-class/multi-label tasks using Softmax

        Args:
            alpha (Union[float, list], optional): Either a float for all channels or
            list of alphas for each channel with length C. Defaults to 0.25.
            gamma (float, optional): Defaults to 2.0.
            reduction (Literal["none", "mean", "sum"], optional): Defaults to "mean".
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_criterion = CrossEntropyLoss(reduction="none")

    def forward(
        self, preds: Tensor, target: Tensor, epoch: int, step: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        ce_loss, _ = self.ce_criterion(preds, target, epoch, step)
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

        return loss, {}
