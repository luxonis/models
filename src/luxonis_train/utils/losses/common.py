import torch
from torch import nn, Tensor
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.n_classes = kwargs.get("n_classes")
        loss_dict = kwargs
        loss_dict.pop("n_classes", None)
        loss_dict.pop("head_attributes", None)
        self.criterion = nn.CrossEntropyLoss(**loss_dict)

    def forward(self, preds, labels, **kwargs):
        if labels.ndim == 4:
            # target should be of size (N,...)
            labels = labels.argmax(dim=1)
        return self.criterion(preds, labels)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.n_classes = kwargs.get("n_classes")
        loss_dict = kwargs
        loss_dict.pop("n_classes", None)
        loss_dict.pop("head_attributes", None)
        self.criterion = nn.BCEWithLogitsLoss(**loss_dict)

    def forward(self, preds, labels, **kwargs):
        return self.criterion(preds, labels)

class SmoothBCELoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, bce_pow: float = 1.0, **_):
        super().__init__()
        self.negative_smooth_const = 1.0 - 0.5 * label_smoothing
        self.positive_smooth_const = 0.5 * label_smoothing
        self.BCE = BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pow]))

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
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
        return self.BCE(prediction, smoothed_target)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        return sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction="mean")


class SegmentationLoss(nn.Module):
    def __init__(self, n_classes, alpha=4.0, gamma=2.0, **kwargs):
        super(SegmentationLoss, self).__init__()

        self.bce = nn.BCELoss(reduction="none")
        self.nc = n_classes
        self.alpha = alpha  # currently not used
        self.gamma = gamma

    def focal_loss(self, logits, labels):
        epsilon = 1.0e-9

        # Focal loss
        fl = -(labels * torch.log(logits + epsilon)) * (1.0 - logits) ** self.gamma
        fl = fl.sum(1)  # Sum focal loss along channel dimension

        # Return mean of the focal loss along spatial
        return fl.mean([1, 2])

    def forward(self, predictions, targets, **kwargs):
        predictions = torch.nn.functional.softmax(predictions, dim=1)

        bs = predictions.shape[0]
        ps = predictions.view(bs, -1)
        ts = targets.view(bs, -1)

        lseg = self.bce(ps, ts.float()).mean(1)

        # focal
        fcl = self.focal_loss(predictions.clone(), targets.clone())

        # iou
        preds = torch.argmax(predictions, dim=1)
        preds = torch.unsqueeze(preds, 1)

        targets = torch.argmax(targets, dim=1)
        masks = torch.unsqueeze(targets, 1)

        ious = torch.zeros(preds.shape[0], device=predictions.device)
        present_classes = torch.zeros(preds.shape[0], device=predictions.device)

        for cls in range(0, self.nc + 1):
            masks_c = masks == cls
            outputs_c = preds == cls
            TP = torch.sum(
                torch.logical_and(masks_c, outputs_c), dim=[1, 2, 3]
            )  # .cpu()
            FP = torch.sum(
                torch.logical_and(torch.logical_not(masks_c), outputs_c), dim=[1, 2, 3]
            )  # .cpu()
            FN = torch.sum(
                torch.logical_and(masks_c, torch.logical_not(outputs_c)), dim=[1, 2, 3]
            )  # .cpu()
            ious += torch.nan_to_num(TP / (TP + FP + FN))
            present_classes += (masks.view(preds.shape[0], -1) == cls).any(
                dim=1
            )  # .cpu()

        iou = ious / present_classes

        liou = 1 - iou

        return (lseg + liou + fcl).mean()
