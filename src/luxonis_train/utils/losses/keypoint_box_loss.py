from collections.abc import Mapping
from typing import Any, Dict, Final, List, Tuple, cast

import torch
from torch import Tensor, nn

from luxonis_train.utils.boxutils import bbox_iou
from luxonis_train.utils.losses.common import BCEWithLogitsLoss

BOX_OFFSET: Final[int] = 5


class KeypointBoxLoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        head_attributes: Mapping[str, Any],
        cls_pw: float = 1.0,
        obj_pw: float = 1.0,
        label_smoothing: float = 0.0,
        iou_ratio: float = 1,
        box_weight: float = 0.05,
        kpt_weight: float = 0.10,
        kptv_weight: float = 0.6,
        cls_weight: float = 0.6,
        obj_weight: float = 0.7,
        anchor_t: float = 4.0,
        **_
    ):
        super().__init__()

        self.n_classes = n_classes

        self.n_keypoints: int = cast(int, head_attributes.get("n_keypoints"))
        self.n_anchors: int = cast(int, head_attributes.get("n_anchors"))
        self.num_heads: int = cast(int, head_attributes.get("num_heads"))
        self.anchors: Tensor = cast(Tensor, head_attributes.get("anchors"))
        if self.num_heads == 3:
            self.balance = [4.0, 1.0, 0.4]
        else:
            self.balance = [4.0, 1.0, 0.25, 0.06, 0.02]

        self.iou_ratio = iou_ratio
        self.box_weight = box_weight
        self.kpt_weight = kpt_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.kptv_weight = kptv_weight
        self.anchor_t = anchor_t

        self.bias = 0.5

        self.BCE_class = BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw]))
        self.BCE_obj = BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))

        # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        self.positive_smooth_const = 1 - 0.5 * label_smoothing
        self.negative_smooth_const = 0.5 * label_smoothing

    def forward(
        self, model_output: Tuple[Tensor, List[Tensor]], targets: Tensor, **_
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Computes the keypoint box loss between the model output
        and the ground truth targets.

        Args:
            model_output (Tuple[Tensor, List[Tensor]]): The model output, which is
                a tuple containing predicted bounding boxes and keypoints and
                a list of the unprocessed outputs from the head. Only the unprocessed
                outputs are used to compute the loss.
                The shapes are
                (batch_size, n_anchors, H, W, 5 + n_classes + n_keypoints * 3)
            targets (Tensor): The ground truth targets, which is a tensor containing the
                batch_index, class labels, bounding box coordinates,
                and keypoint coordinates for each object in the image.
                The shape is (n_objects, 5 + n_classes + n_keypoints * 2)

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: A tuple containing the total loss and a
                dictionary of sub-losses for each component of the loss
                (box, objectness, class, keypoints, keypoints visibility).
        """
        predictions = model_output[1]
        device = predictions[0].device
        sub_losses = {
            "box": torch.zeros(1, device=device),
            "obj": torch.zeros(1, device=device),
            "cls": torch.zeros(1, device=device),
            "kptv": torch.zeros(1, device=device),
            "kpt": torch.zeros(1, device=device),
        }

        (
            class_targets,
            box_targets,
            kpt_targets,
            indices,
            anchors,
        ) = self._construct_targets(predictions, targets)

        targets = targets.to(device)

        for i, pred in enumerate(predictions):
            batch_index, anchor_index, grid_y, grid_x = indices[i]
            obj_targets = torch.zeros_like(pred[..., 0], device=device)

            n_targets = len(batch_index)
            if n_targets != 0:
                pred_subset = pred[batch_index, anchor_index, grid_y, grid_x]

                box_loss, iou = self._compute_box_loss(
                    pred_subset[:, :4], anchors[i], box_targets[i]
                )
                sub_losses["box"] += box_loss

                kpt_loss, kpt_visibility_loss = self._compute_keypoint_loss(
                    pred_subset, kpt_targets[i].to(device)
                )

                sub_losses["kpt"] += kpt_loss
                sub_losses["kptv"] += kpt_visibility_loss

                obj_targets[batch_index, anchor_index, grid_y, grid_x] = (
                    1.0 - self.iou_ratio
                ) + self.iou_ratio * iou.detach().clamp(0).to(obj_targets.dtype)

                if self.n_classes > 1:
                    sub_losses["cls"] += self._compute_class_loss(
                        pred_subset[:, BOX_OFFSET : BOX_OFFSET + self.n_classes],
                        class_targets[i],
                    )

            sub_losses["obj"] += (
                self._compute_objectness_loss(pred, obj_targets) * self.balance[i]
            )

        loss = cast(Tensor, sum(sub_losses.values())).reshape([])
        return loss, {name: loss.detach() for name, loss in sub_losses.items()}

    def _compute_objectness_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Computes the objectness loss for the given prediction and target tensors.

        Args:
            prediction (Tensor): The predicted tensor of shape
            (batch_size, n_anchors, H, W, 5 + n_classes + n_keypoints * 3)
            target (Tensor): The target tensor of shape
                (batch_size, n_anchors, H, W)

        Returns:
            Tensor: The computed objectness loss.
        """
        return self.BCE_obj(prediction[..., 4], target) * self.obj_weight

    def _compute_class_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Computes the classification loss for the predicted bounding boxes.

        Args:
            prediction (torch.Tensor): A tensor of shape (n_detections, n_classes),
                containing the predicted class scores and box coordinates for each box.
            target (torch.Tensor): A tensor of shape (n_detections, ), containing the
                ground-truth class labels for each box.

        Returns:
            torch.Tensor: A scalar tensor representing the classification loss.
        """
        class_target = torch.full_like(
            prediction,
            self.negative_smooth_const,
            device=prediction.device,
        )
        class_target[torch.arange(len(target)), target] = self.positive_smooth_const
        return self.BCE_class(prediction, class_target) * self.cls_weight

    def _compute_keypoint_loss(
        self, prediction: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the keypoint loss and visibility loss
        for a given prediction and target.

        Args:
            prediction (Tensor): The predicted tensor of shape
                (n_detections, (5 + n_classes + n_keypoints * 3)).
            target (Tensor): The target tensor of shape (n_detections, n_keypoints * 2).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the keypoint loss
            tensor of shape (1,) and the visibility loss tensor of shape (1,).
        """
        x = prediction[:, BOX_OFFSET + self.n_classes :: 3] * 2.0 - 0.5
        y = prediction[:, BOX_OFFSET + self.n_classes + 1 :: 3] * 2.0 - 0.5
        visibility_score = prediction[:, BOX_OFFSET + self.n_classes + 2 :: 3]

        mask = target[:, 0::2] != 0
        visibility_loss = self.BCE_class(visibility_score, mask.float())
        distance = (x - target[:, 0::2]) ** 2 + (y - target[:, 1::2]) ** 2

        loss_factor = (torch.sum(mask != 0) + torch.sum(mask == 0)) / (
            torch.sum(mask != 0) + 1e-9
        )
        loss = loss_factor * (torch.log(distance + 1 + 1e-9) * mask).mean()
        return loss * self.kpt_weight, visibility_loss * self.kptv_weight

    def _compute_box_loss(
        self, prediction: Tensor, anchor: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the box loss for a batch of predictions, anchors and targets.

        Args:
            prediction (Tensor): A tensor of shape (n_detections, 4) containing
                the predicted bounding box coordinates (x, y, w, h).
            anchor (Tensor): A tensor of shape (n_detections, 2) containing the anchor
                box coordinates (w, h).
            target (Tensor): A tensor of shape (n_detections, 4) containing the
                target bounding box coordinates (x, y, w, h).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the box loss and the IoU
            between the predicted and target boxes. The box loss is a tensor of
            shape (1,) and the IoU is a tensor of shape (n_detections,).
        """
        device = prediction.device
        x_y = prediction[:, :2].sigmoid() * 2.0 - 0.5
        w_h = (prediction[:, 2:].sigmoid() * 2) ** 2 * anchor.to(device)
        boxes = torch.cat((x_y, w_h), 1).T
        iou = bbox_iou(boxes, target.to(device), box_format="xywh", iou_type="ciou")
        return (1.0 - iou).mean() * self.box_weight, iou

    def _construct_targets(
        self, predictions: List[Tensor], targets: Tensor
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        List[Tensor],
    ]:
        """
        Constructs targets for computation of the individual sub-losses.

        Args:
            predictions (List[Tensor]]): A list of the unprocessed outputs
                from the head. The shapes are
                (batch_size, n_anchors, H, W, 5 + n_classes + n_keypoints * 3)
            targets (Tensor): The ground truth targets, which is a tensor containing the
                batch_index, class labels, bounding box coordinates,
                and keypoint coordinates for each object in the image.
                The shape is (n_objects, 5 + n_classes + n_keypoints * 2)
        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor],
            List[Tuple[Tensor, Tensor, Tensor, Tensor]], List[Tensor]]: A tuple
                containing the class targets, box targets,
                keypoint targets, indices, and anchors.
        """
        n_targets = len(targets)

        class_targets: List[Tensor] = []
        box_targets: List[Tensor] = []
        keypoint_targets: List[Tensor] = []
        indices: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
        anchors: List[Tensor] = []

        gain_length = 2 * self.n_keypoints + BOX_OFFSET + 2
        gain = torch.ones(gain_length, device=targets.device)
        anchor_indices = (
            torch.arange(self.n_anchors, device=targets.device, dtype=torch.float32)
            .reshape(self.n_anchors, 1)
            .repeat(1, n_targets)
            .unsqueeze(-1)
        )
        targets = torch.cat((targets.repeat(self.n_anchors, 1, 1), anchor_indices), 2)

        offsets = (
            torch.tensor(
                [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device
            ).float()
            * self.bias
        )

        for i in range(self.num_heads):
            anchor = self.anchors[i]
            gain[2 : gain_length - 1] = torch.tensor(predictions[i].shape)[
                (2 + self.n_keypoints) * [3, 2]
            ]
            target_gain, offset = self._match_to_anchors(
                targets, anchor, gain, offsets, n_targets
            )

            batch_index, cls = target_gain[:, :2].long().T
            grid_xy = target_gain[:, 2:4]
            grid_wh = target_gain[:, 4:6]
            grid_delta = (grid_xy - offset).long()
            grid_delta_x, grid_delta_y = grid_delta.T

            anchor_indices = target_gain[:, -1].long()
            indices.append(
                (
                    batch_index,
                    anchor_indices,
                    grid_delta_y.clamp_(0, gain[3].long() - 1),  # type: ignore
                    grid_delta_x.clamp_(0, gain[2].long() - 1),  # type: ignore
                )
            )
            box_targets.append(torch.cat((grid_xy - grid_delta, grid_wh), 1))
            for kpt in range(self.n_keypoints):
                low = BOX_OFFSET + 1 + 2 * kpt
                high = BOX_OFFSET + 1 + 2 * (kpt + 1)
                target_gain[:, low:high][target_gain[:, low:high] != 0] -= grid_delta[
                    target_gain[:, low:high] != 0
                ]
            keypoint_targets.append(target_gain[:, BOX_OFFSET + 1 : -1])
            anchors.append(anchor[anchor_indices])
            class_targets.append(cls)

        return class_targets, box_targets, keypoint_targets, indices, anchors

    def _match_to_anchors(
        self,
        targets: Tensor,
        anchor: Tensor,
        gain: Tensor,
        offsets: Tensor,
        n_targets: int,
    ) -> Tuple[Tensor, Tensor]:
        target_gain = targets * gain
        if n_targets == 0:
            return targets[0], torch.zeros(1, device=targets.device)
        ratio = target_gain[:, :, 4:6] / anchor.unsqueeze(1)
        target_gain = target_gain[
            torch.max(ratio, 1.0 / ratio).max(2)[0] < self.anchor_t
        ]

        grid_xy = target_gain[:, 2:4]
        grid_inverse = gain[[2, 3]] - grid_xy
        mask = self._construct_grid_mask(grid_xy, grid_inverse, self.bias)
        target_gain = target_gain.repeat((5, 1, 1))[mask]
        offset = (torch.zeros_like(grid_xy).unsqueeze(0) + offsets.unsqueeze(1))[mask]
        return target_gain, offset

    def _construct_grid_mask(
        self, grid_xy: Tensor, grid_inverse: Tensor, bias: float
    ) -> Tensor:
        def decimal_part(x: Tensor) -> Tensor:
            return x % 1.0

        x, y = ((decimal_part(grid_xy) < bias) & (grid_xy > 1.0)).T
        j, k = ((decimal_part(grid_inverse) < bias) & (grid_inverse > 1.0)).T
        return torch.stack((torch.ones_like(x), x, y, j, k))
