import torch
from torch import Tensor

from luxonis_train.utils.types import Labels, LabelType, ModulePacket

from .common import (
    draw_segmentation_labels,
    draw_segmentation_masks,
    seg_output_to_bool,
)
from .luxonis_visualizer import LuxonisVisualizer


class SegmentationVisualizer(LuxonisVisualizer):
    def forward(
        self, img: Tensor, output: ModulePacket, label: Labels, idx: int = 0
    ) -> Tensor:
        prediction = output["segmentation"][0][idx]
        masks = seg_output_to_bool(prediction)
        masks = masks.cpu()
        orig_img = img.clone()
        img_labels = draw_segmentation_labels(
            orig_img, label[LabelType.SEGMENTATION][0][idx]
        )

        img = img.cpu()
        img = draw_segmentation_masks(img, masks, alpha=0.4)
        return torch.cat([img_labels, img], dim=-1)
