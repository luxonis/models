from typing import Literal

from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from luxonis_train.attached_modules.losses import BaseLoss


class SigmoidFocalLoss(BaseLoss[Tensor, Tensor]):
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

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = sigmoid_focal_loss(
            preds, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )

        return loss