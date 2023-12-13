import logging

import torch
from torch import Tensor

from luxonis_train.utils.types import BBoxProtocol, LabelType

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_bounding_box_labels,
    draw_bounding_boxes,
    get_color,
)


class BBoxVisualizer(BaseVisualizer[list[Tensor], Tensor]):
    """Visualizer for bounding box predictions.

    Creates a visualization of the bounding box predictions and labels.
    """

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        colors: dict[str, Color] | list[Color] | None = None,
        fill: bool = False,
        width: int | None = None,
        font: str | None = None,
        font_size: int | None = None,
        **kwargs,
    ):
        """Constructor for the BBoxVisualizer module.

        Args:
            labels (dict[int, str] | list[str], optional): Either a dictionary mapping
              class indices to names, or a list of names. If list is provided, the
              label mapping is done by index. By default, no labels are drawn.
            colors (dict[int, Color] | list[Color], optional):
              Either a dictionary mapping class indices to colors, or a list of colors.
              If list is provided, the color mapping is done by index.
              By default, random colors are used.
            fill (bool, optional): Whether or not to fill the bounding boxes.
              Defaults to False.
            width (int, optional): The width of the bounding box lines. Defaults to 1.
            font (str, optional): A filename containing a TrueType font.
              Defaults to None.
            font_size (int, optional): The font size to use for the labels.
              Defaults to None.
        """
        super().__init__(
            required_labels=[LabelType.BOUNDINGBOX], protocol=BBoxProtocol, **kwargs
        )
        if isinstance(labels, list):
            labels = {i: label for i, label in enumerate(labels)}

        self.labels = labels or {
            i: label for i, label in enumerate(self.node.class_names)
        }
        if colors is None:
            colors = {label: get_color(i) for i, label in self.labels.items()}
        if isinstance(colors, list):
            colors = {self.labels[i]: color for i, color in enumerate(colors)}
        self.colors = colors
        self.fill = fill
        self.width = width
        self.font = font
        self.font_size = font_size
        self.draw_labels = draw_labels

    @staticmethod
    def draw_targets(
        canvas: Tensor, targets: Tensor, width: int | None = None, **kwargs
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            target = targets[targets[:, 0] == i]

            *_, H, W = canvas.shape
            width = width or max(1, int(min(H, W) / 100))
            viz[i] = draw_bounding_box_labels(
                canvas[i].clone(),
                target[:, 2:],
                width=width,
                **kwargs,
            ).to(canvas.device)

        return viz

    @staticmethod
    def draw_predictions(
        canvas: Tensor, predictions: list[Tensor], width: int | None = None, **kwargs
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            prediction = predictions[i]

            *_, H, W = canvas.shape
            width = width or max(1, int(min(H, W) / 100))
            try:
                viz[i] = draw_bounding_boxes(
                    canvas[i].clone(),
                    prediction[:, :4],
                    width=width,
                    **kwargs,
                )
            except ValueError as e:
                logging.getLogger(__name__).warning(
                    f"Failed to draw bounding boxes: {e}. Skipping visualization."
                )
                viz = canvas
        return viz

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        predictions: list[Tensor],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Creates a visualization of the bounding box predictions and labels.

        Args:
            label_canvas (Tensor): The canvas containing the labels.
            prediction_canvas (Tensor): The canvas containing the predictions.
            prediction (Tensor): The predicted bounding boxes. The shape should be
              [N, 6], where N is the number of bounding boxes and the last dimension
              is [x1, y1, x2, y2, class, conf].
            targets (Tensor): The target bounding boxes.

        Returns:
            tuple[Tensor, Tensor]: A tuple of the label and prediction visualizations.
        """
        target_classes = targets[..., 1].int()
        target_labels = (
            [self.labels[int(c)] for c in target_classes]
            if self.draw_labels and self.labels
            else None
        )
        target_colors = (
            [self.colors[self.labels[int(c)]] for c in target_classes]
            if self.colors
            else None
        )
        prediction_classes = predictions[0][..., 5].int()
        pred_labels = (
            [self.labels[int(c)] for c in prediction_classes]
            if self.draw_labels and self.labels
            else None
        )
        pred_colors = (
            [self.colors[self.labels[int(c)]] for c in prediction_classes]
            if self.colors
            else None
        )
        targets_viz = self.draw_targets(
            label_canvas,
            targets,
            labels=target_labels,
            colors=target_colors,
            fill=self.fill,
            font=self.font,
            font_size=self.font_size,
            width=self.width,
        )
        predictions_viz = self.draw_predictions(
            prediction_canvas, predictions, labels=pred_labels, colors=pred_colors
        )
        return targets_viz, predictions_viz.to(targets_viz.device)
