from typing import Dict, List, Tuple, cast

import torch
from luxonis_ml.loader import LabelType
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.losses.box_loss import BoxLoss
from luxonis_train.losses.keypoint_loss import KeypointLoss
from luxonis_train.losses.luxonis_loss import LuxonisLoss
from luxonis_train.utils.boxutils import match_to_anchor
from luxonis_train.utils.types import Labels, ModulePacket

from .common import BCEWithLogitsLoss, SmoothBCEWithLogitsLoss

KeypointTargetType = Tuple[
    List[Tensor],
    List[Tensor],
    List[Tensor],
    List[Tuple[Tensor, Tensor, Tensor, Tensor]],
    List[Tensor],
]


class KeypointBoxLoss(LuxonisLoss[list[Tensor], KeypointTargetType]):
    """Joint loss for keypoint and box predictions for cases where
    the keypoints and boxes are inherently linked.
    Based on `YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss`,
    https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf
    """

    def validate(self, outputs: ModulePacket, labels: Labels) -> None:
        assert "features" in outputs, "Outputs must contain 'features' key."
        assert LabelType.BOUNDINGBOX in labels, "Labels must contain 'boundingbox' key."
        assert LabelType.KEYPOINT in labels, "Labels must contain 'keypoint' key."

    def preprocess(
        self, outputs: ModulePacket, label_dict: Labels
    ) -> tuple[list[Tensor], KeypointTargetType]:
        """
        Posptrocesses the labels to be in the correct format for loss calculation.
        Args:
                output (Tuple[Tensor, List[Tensor]]): Output from the forward pass.
                label_dict (Dict[str, Tensor]): Dictionary containing the labels.
        Returns:
                Tuple[Tuple[Tensor,
                            List[Tensor]],
                            Tuple[
                                List[Tensor],
                                List[Tensor],
                                List[Tensor],
                                List[Tuple[Tensor, Tensor, Tensor, Tensor]],
                                List[Tensor]]]:
                        Tuple containing the original output and the postprocessed
                        labels.
                        The processed labels are a tuple containing the class targets,
                        box targets, keypoint targets, indices and anchors.
                        Indicies are a tuple containing vectors of indices for batch,
                        anchor, feature y and feature x dimensions, respectively.
                        They are all of shape (n_targets,).
                        The indices are used to index the output tensors of shape
                        (batch_size, n_anchors, feature_height, feature_width,
                        n_classes + box_offset + n_keypoints * 3) to get a tensor of
                        shape (n_targets, n_classes + box_offset + n_keypoints * 3).
        """
        predictions = outputs["features"]
        kpts = label_dict[LabelType.KEYPOINT]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        targets = torch.zeros((len(boxes), nkpts * 2 + self.box_offset + 1))
        targets[:, :2] = boxes[:, :2]
        targets[:, 2 : self.box_offset + 1] = box_convert(
            boxes[:, 2:], "xywh", "cxcywh"
        )
        targets[:, self.box_offset + 1 :: 2] = kpts[:, 2::3]  # insert kp x coordinates
        targets[:, self.box_offset + 2 :: 2] = kpts[:, 3::3]  # insert kp y coordinates

        n_targets = len(targets)

        class_targets: List[Tensor] = []
        box_targets: List[Tensor] = []
        keypoint_targets: List[Tensor] = []
        indices: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
        anchors: List[Tensor] = []

        anchor_indices = (
            torch.arange(self.n_anchors, device=targets.device, dtype=torch.float32)
            .reshape(self.n_anchors, 1)
            .repeat(1, n_targets)
            .unsqueeze(-1)
        )
        targets = torch.cat((targets.repeat(self.n_anchors, 1, 1), anchor_indices), 2)

        XY_SHIFTS = (
            torch.tensor(
                [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device
            ).float()
            * self.bias
        )

        for i in range(self.num_heads):
            anchor = self.anchors[i]
            feature_height, feature_width = predictions[i].shape[2:4]

            scaled_targets, xy_shifts = match_to_anchor(
                targets,
                anchor,
                XY_SHIFTS,
                feature_width,
                feature_height,
                self.n_keypoints,
                self.anchor_threshold,
                self.bias,
                self.box_offset,
            )

            batch_index, cls = scaled_targets[:, :2].long().T
            box_xy = scaled_targets[:, 2:4]
            box_wh = scaled_targets[:, 4:6]
            box_xy_deltas = (box_xy - xy_shifts).long()
            feature_x_index = box_xy_deltas[:, 0].clamp_(0, feature_width - 1)
            feature_y_index = box_xy_deltas[:, 1].clamp_(0, feature_height - 1)

            anchor_indices = scaled_targets[:, -1].long()
            indices.append(
                (
                    batch_index,
                    anchor_indices,
                    feature_y_index,
                    feature_x_index,
                )
            )
            class_targets.append(cls)
            box_targets.append(torch.cat((box_xy - box_xy_deltas, box_wh), 1))
            anchors.append(anchor[anchor_indices])

            keypoint_targets.append(
                self._create_keypoint_target(scaled_targets, box_xy_deltas)
            )

        return predictions, (
            class_targets,
            box_targets,
            keypoint_targets,
            indices,
            anchors,
        )

    def _create_keypoint_target(self, scaled_targets: Tensor, box_xy_deltas: Tensor):
        keypoint_target = scaled_targets[:, self.box_offset + 1 : -1]
        for j in range(self.n_keypoints):
            low = 2 * j
            high = 2 * (j + 1)
            keypoint_mask = keypoint_target[:, low:high] != 0
            keypoint_target[:, low:high][keypoint_mask] -= box_xy_deltas[keypoint_mask]
        return keypoint_target

    def __init__(
        self,
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
        bias: float = 0.5,
        balance: List[float] = [4.0, 1.0, 0.4],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_classes: int = cast(int, self.module_attributes.get("n_classes"))
        self.n_keypoints: int = cast(int, self.module_attributes.get("n_keypoints"))
        self.n_anchors: int = cast(int, self.module_attributes.get("n_anchors"))
        self.num_heads: int = cast(int, self.module_attributes.get("num_heads"))
        self.box_offset: int = cast(int, self.module_attributes.get("box_offset"))
        self.anchors: Tensor = cast(Tensor, self.module_attributes.get("anchors"))
        self.balance = balance
        if len(self.balance) < self.num_heads:
            raise ValueError(
                f"Balance list must have at least {self.num_heads} elements."
            )

        self.iou_ratio = iou_ratio
        self.box_weight = box_weight
        self.kpt_weight = kpt_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.kptv_weight = kptv_weight
        self.anchor_threshold = anchor_t

        self.bias = bias

        self.BCE = BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))
        self.class_loss = SmoothBCEWithLogitsLoss(
            label_smoothing=label_smoothing, bce_pow=cls_pw
        )
        self.keypoint_loss = KeypointLoss(bce_power=cls_pw)
        self.box_loss = BoxLoss()

        self.positive_smooth_const = 1 - 0.5 * label_smoothing
        self.negative_smooth_const = 0.5 * label_smoothing

    def compute_loss(
        self,
        predictions: List[Tensor],
        targets: KeypointTargetType,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = predictions[0].device
        sub_losses = {
            "box": torch.zeros(1, device=device),
            "obj": torch.zeros(1, device=device),
            "cls": torch.zeros(1, device=device),
            "kptv": torch.zeros(1, device=device),
            "kpt": torch.zeros(1, device=device),
        }

        for pred, class_target, box_target, kpt_target, index, anchor, balance in zip(
            predictions, *targets, self.balance
        ):
            obj_targets = torch.zeros_like(pred[..., 0], device=device)

            n_targets = len(class_target)
            if n_targets > 0:
                pred_subset = pred[index]

                box_loss, iou = self.box_loss(pred_subset[:, :4], anchor, box_target)
                sub_losses["box"] += box_loss * self.box_weight

                kpt_loss, kpt_visibility_loss = self.keypoint_loss(
                    pred_subset[:, self.box_offset + self.n_classes :],
                    kpt_target.to(device),
                )

                sub_losses["kpt"] += kpt_loss * self.kpt_weight
                sub_losses["kptv"] += kpt_visibility_loss * self.kptv_weight

                obj_targets[index] = (
                    1.0 - self.iou_ratio
                ) + self.iou_ratio * iou.detach().clamp(0).to(obj_targets.dtype)

                if self.n_classes > 1:
                    sub_losses["cls"] += (
                        self.class_loss.compute_loss(
                            [
                                pred_subset[
                                    :,
                                    self.box_offset : self.box_offset + self.n_classes,
                                ]
                            ],
                            class_target,
                        )[0]
                        * self.cls_weight
                    )

            sub_losses["obj"] += (
                self.BCE.compute_loss(pred[..., 4], obj_targets)[0]
                * balance
                * self.obj_weight
            )

        loss = cast(Tensor, sum(sub_losses.values())).reshape([])
        return loss, {name: loss.detach() for name, loss in sub_losses.items()}
