from torch import Tensor

from luxonis_train.utils.types import Labels, LabelType, Packet, SegmentationProtocol

from .luxonis_visualizer import LuxonisVisualizer
from .utils import (
    draw_segmentation_labels,
    draw_segmentation_masks,
    seg_output_to_bool,
)


class SegmentationVisualizer(LuxonisVisualizer[Tensor, Tensor]):
    def __init__(self, color: str = "#5050FF", alpha: float = 0.6, **kwargs):
        super().__init__(
            protocol=SegmentationProtocol,
            required_labels=[LabelType.SEGMENTATION],
            **kwargs,
        )
        self.color = color
        self.alpha = alpha

    def prepare(self, output: Packet[Tensor], label: Labels) -> tuple[Tensor, Tensor]:
        return output["segmentation"][0], label[LabelType.SEGMENTATION]

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        idx: int,
        prediction: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        masks = seg_output_to_bool(prediction[idx])
        masks = masks.to(prediction_canvas.device)
        label_viz = draw_segmentation_labels(
            label_canvas.clone(),
            targets[idx],
            alpha=self.alpha,
            colors=self.color,
        ).to(label_canvas.device)

        prediction_viz = draw_segmentation_masks(
            prediction_canvas.clone(), masks, alpha=self.alpha, colors=self.color
        )
        return label_viz, prediction_viz
