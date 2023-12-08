from torch import Tensor

from luxonis_train.utils.types import (
    Labels,
    LabelType,
    Packet,
)

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_keypoint_labels,
    draw_keypoints,
)


class KeypointVisualizer(BaseVisualizer[list[Tensor], Tensor]):
    def __init__(
        self,
        visibility_threshold: float = 0.5,
        connectivity: list[tuple[int, int]] | None = None,
        visible_color: Color = "red",
        nonvisible_color: Color | None = None,
        **kwargs,
    ):
        """

        Args:
            visibility_threshold (float): Threshold for visibility of keypoints.
              If the visibility of a keypoint is below this threshold, it is
              considered as not visible. Defaults to 0.5.
            connectivity (list[tuple[int, int]] | None): List of tuples of
              keypoint indices that define the connections in the skeleton.
              Defaults to None.
            visible_color (Color): Color of visible keypoints.
              Either a string or a tuple of RGB values. Defaults to "red".
            nonvisible_color (Color | None): Color of nonvisible keypoints.
              If None, nonvisible keypoints are not drawn. Defaults to None.
        """
        super().__init__(required_labels=[LabelType.KEYPOINT], **kwargs)
        self.visibility_threshold = visibility_threshold
        self.connectivity = connectivity
        self.visible_color = visible_color
        self.nonvisible_color = nonvisible_color

    def prepare(
        self, output: Packet[Tensor], label: Labels
    ) -> tuple[list[Tensor], Tensor]:
        return output["keypoints"], label[LabelType.KEYPOINT]

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        idx: int,
        predictions: list[Tensor],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prediction = predictions[idx][:, 1:]
        targets = targets[targets[:, 0] == idx][:, 1:]
        mask = prediction[..., 2] < self.visibility_threshold
        visible_kpts = prediction[..., :2] * (~mask).unsqueeze(-1).float()
        prediction_viz = draw_keypoints(
            prediction_canvas.clone(),
            visible_kpts[..., :2],
            colors=self.visible_color,
            connectivity=self.connectivity,
        )
        if self.nonvisible_color is not None:
            nonvisible_kpts = prediction[..., :2] * mask.unsqueeze(-1).float()
            prediction_viz = draw_keypoints(
                prediction_viz,
                nonvisible_kpts[..., :2],
                colors=self.nonvisible_color,
                connectivity=self.connectivity,
            )
        label_viz = draw_keypoint_labels(
            label_canvas.clone(),
            targets,
            colors=self.visible_color,
            connectivity=self.connectivity,
        )

        return label_viz, prediction_viz
