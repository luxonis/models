import torch
from torch import Tensor

from luxonis_train.utils.boxutils import non_max_suppression
from luxonis_train.utils.types import Labels, LabelType, ModulePacket

from .common import (
    draw_bounding_box_labels,
    draw_bounding_boxes,
    draw_keypoint_labels,
    draw_keypoints,
)
from .luxonis_visualizer import LuxonisVisualizer


class KeypointVisualizer(LuxonisVisualizer):
    def __init__(
        self,
        n_keypoints: int,
        n_classes: int,
        visibility_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_keypoints = n_keypoints
        self.n_classes = n_classes
        self.visibility_threshold = visibility_threshold
        self.connectivity = None

    def forward(
        self, img: Tensor, output: ModulePacket, label: Labels, idx: int = 0
    ) -> Tensor:
        orig_img = img.clone()
        prediction = output["keypoints"][0][idx]

        nms = non_max_suppression(
            prediction.unsqueeze(0),
            n_classes=self.n_classes,
            conf_thres=0.25,
            iou_thres=0.45,
            box_format="cxcywh",
        )[0]
        bboxes = nms[:, :4]
        img = draw_bounding_boxes(img, bboxes)
        kpts = nms[:, 6:].reshape(-1, self.n_keypoints, 3)
        mask = kpts[:, :, 2] < self.visibility_threshold
        kpts = kpts[:, :, 0:2] * (~mask).unsqueeze(-1).float()
        img = draw_keypoints(
            img, kpts[..., :2], colors="red", connectivity=self.connectivity
        )
        kpt_label = label[LabelType.KEYPOINT]
        kpt_label = kpt_label[kpt_label[:, 0] == idx][:, 1:]

        box_label = label[LabelType.BOUNDINGBOX]
        box_label = box_label[box_label[:, 0] == idx]
        img_with_labels = draw_keypoint_labels(orig_img, kpt_label)
        img_with_labels = draw_bounding_box_labels(img_with_labels, box_label)

        return torch.cat([img_with_labels, img], dim=-1)
