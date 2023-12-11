import torch
from torch import Tensor

from luxonis_train.utils.types import Labels, LabelType, Packet, SegmentationProtocol

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_segmentation_labels,
    draw_segmentation_masks,
    seg_output_to_bool,
)


class SegmentationVisualizer(BaseVisualizer[Tensor, Tensor]):
    def __init__(self, color: Color = "#5050FF", alpha: float = 0.6, **kwargs):
        """

        Args:
            color (str): Color of the segmentation masks. Defaults to "#5050FF".
            alpha (float): Alpha value of the segmentation masks. Defaults to 0.6.
        """
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
        predictions: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Creates a visualization of the segmentation predictions and labels.

        Args:
            label_canvas (Tensor): The canvas to draw the labels on.
            prediction_canvas (Tensor): The canvas to draw the predictions on.
            predictions (Tensor): The predictions to visualize.
            targets (Tensor): The targets to visualize.

        Returns:
            tuple[Tensor, Tensor]: A tuple of the label and prediction visualizations.
        """
        labels_viz = torch.zeros_like(label_canvas)
        predictions_viz = torch.zeros_like(prediction_canvas)
        for i in range(len(labels_viz)):
            prediction = predictions[i]
            target = targets[i]
            mask = seg_output_to_bool(prediction)
            mask = mask.to(prediction_canvas.device)
            labels_viz[i] = draw_segmentation_labels(
                label_canvas[i].clone(),
                target,
                alpha=self.alpha,
                colors=self.color,
            ).to(label_canvas.device)

            predictions_viz[i] = draw_segmentation_masks(
                prediction_canvas[i].clone(), mask, alpha=self.alpha, colors=self.color
            )
        return labels_viz, predictions_viz
