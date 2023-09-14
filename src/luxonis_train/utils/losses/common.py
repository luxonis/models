import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal, Union, List
from torchvision.ops import sigmoid_focal_loss

from luxonis_train.utils.losses.base_loss import BaseLoss


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

        if self.head_attributes.get("n_classes") == 1:
            raise ValueError(
                f"`{self.get_name()}` should be only used for multi-class/multi-label tasks"
            )

        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, preds: Tensor, target: Tensor, epoch: int, step: int) -> Tensor:
        if target.ndim == 4:
            # target should be of size (N,...)
            target = target.argmax(dim=1)
        return self.criterion(preds, target)


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

        if self.head_attributes.get("n_classes") != 1:
            raise ValueError(
                f"`{self.get_name()}` should be only used for binary tasks"
            )

        self.criterion = nn.BCEWithLogitsLoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )

    def forward(self, preds: Tensor, target: Tensor, epoch: int, step: int) -> Tensor:
        return self.criterion(preds, target)


class BinaryFocalLoss(BaseLoss):
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

        if self.head_attributes.get("n_classes") != 1:
            raise ValueError(
                f"`{self.get_name()}` should be only used for binary tasks"
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: Tensor, target: Tensor, epoch: int, step: int) -> Tensor:
        loss = sigmoid_focal_loss(
            preds, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )

        return loss


class MultiClassFocalLoss(BaseLoss):
    def __init__(
        self,
        alpha: Union[float, List[float]] = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Focal loss implementation for multi-class/multi-label tasks

        Args:
            alpha (Union[float, list], optional): Either a float for all channels or
            list of alphas for each channel with length C. Defaults to 0.25.
            gamma (float, optional): Defaults to 2.0.
            reduction (Literal["none", "mean", "sum"], optional): Defaults to "mean".
        """
        super().__init__(**kwargs)

        if self.head_attributes.get("n_classes") == 1:
            raise ValueError(
                f"`{self.get_name()}` should be only used for multi-class/multi-label tasks"
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: Tensor, target: Tensor, epoch: int, step: int) -> Tensor:
        if target.ndim == 4:
            # target should be of size (N,...)
            target = target.argmax(dim=1)

        ce_loss = F.cross_entropy(preds, target, reduction="none")
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
