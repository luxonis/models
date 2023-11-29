import torch
from torch import Tensor

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss


class SmoothBCEWithLogitsLoss(BaseLoss[list[Tensor], Tensor]):
    """BCE with logits loss with label smoothing.

    Args:
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0.
        bce_pow (float, optional): Weight for positive samples. Defaults to 1.0.
    """

    def __init__(self, label_smoothing: float = 0.0, bce_pow: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.negative_smooth_const = 1.0 - 0.5 * label_smoothing
        self.positive_smooth_const = 0.5 * label_smoothing
        self.criterion = BCEWithLogitsLoss(
            node_attributes=self.node_attributes, pos_weight=torch.tensor([bce_pow])
        )

    def forward(self, predictions: list[Tensor], target: Tensor) -> Tensor:
        """Computes the BCE loss with label smoothing.

        Args:
            prediction (Tensor): A tensor of shape (N, n_classes),
                containing the predicted class scores.
            target (Tensor): A tensor of shape (N,), containing the
                ground-truth class labels

        Returns:
            Tensor: A scalar tensor.
        """
        prediction = predictions[0]
        smoothed_target = torch.full_like(
            prediction,
            self.negative_smooth_const,
            device=prediction.device,
        )
        smoothed_target[torch.arange(len(target)), target] = self.positive_smooth_const
        return self.criterion.forward(prediction, smoothed_target)
