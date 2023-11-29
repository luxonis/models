import warnings

from torch import Tensor

from luxonis_train.utils.types import BBoxProtocol, LabelType

from .base_visualizer import BaseVisualizer
from .utils import (
    draw_bounding_box_labels,
    draw_bounding_boxes,
    get_color,
)


class BBoxVisualizer(BaseVisualizer[Tensor, Tensor]):
    """Visualizer for bounding box predictions.

    Creates a visualization of the bounding box predictions and labels.
    """

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        colors: dict[str, tuple[int, int, int] | str]
        | list[tuple[int, int, int] | str]
        | None = None,
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
              label mapping is done by index. By default, no labels are used.
            colors (dict[int, tuple[int, int, int] | str]
                    | list[tuple[int, int, int] | str], optional):
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
            i: label
            for i, label in enumerate(self.node_attributes.dataset_metadata.class_names)
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

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        idx: int,
        prediction: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Creates a visualization of the bounding box predictions and labels.

        Args:
            label_canvas (Tensor): The canvas containing the labels.
            prediction_canvas (Tensor): The canvas containing the predictions.
            prediction (Tensor): The predicted bounding boxes. The shape should be
              [N, 6], where N is the number of bounding boxes and the last dimension
              is [x1, y1, x2, y2].
            targets (Tensor): The target bounding boxes.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the canvas with the labels
            and the canvas with the predictions.
        """
        prediction = prediction[idx]
        targets = targets[targets[:, 0] == idx]
        prediction_classes = prediction[:, 5].int()

        target_classes = targets[:, 1].int()
        *_, H, W = label_canvas.shape
        width = self.width or max(1, int(min(H, W) / 100))
        labels_viz = draw_bounding_box_labels(
            label_canvas.clone(),
            targets[:, 2:],
            labels=[self.labels[int(c)] for c in target_classes]
            if self.draw_labels and self.labels
            else None,
            colors=[self.colors[self.labels[int(c)]] for c in target_classes]
            if self.colors
            else None,
            fill=self.fill,
            width=width,
            font=self.font,
            font_size=self.font_size,
        ).to(prediction_canvas.device)

        try:
            prediction_viz = draw_bounding_boxes(
                prediction_canvas.clone(),
                prediction[:, :4],
                labels=[self.labels[int(c)] for c in prediction_classes]
                if self.draw_labels and self.labels
                else None,
                colors=[self.colors[self.labels[int(c)]] for c in prediction_classes]
                if self.colors
                else None,
                fill=self.fill,
                width=width,
                font=self.font,
                font_size=self.font_size,
            )
        except ValueError as e:
            warnings.warn(
                f"Failed to draw bounding boxes: {e}. Skipping visualization."
            )
            prediction_viz = prediction_canvas
        return labels_viz, prediction_viz.to(labels_viz.device)
