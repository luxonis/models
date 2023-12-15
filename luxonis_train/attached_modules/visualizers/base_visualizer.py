from abc import abstractmethod

from torch import Tensor
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.utils.registry import VISUALIZERS
from luxonis_train.utils.types import Labels, Packet

Ts = TypeVarTuple("Ts")


class BaseVisualizer(
    BaseAttachedModule[Unpack[Ts]],
    register=False,
    registry=VISUALIZERS,
):
    """A base class for all visualizers.

    This class defines the basic interface for all visualizers.
    It utilizes automatic registration of defined subclasses
    to the `VISUALIZERS` registry.
    """

    @abstractmethod
    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        *args: Unpack[Ts],
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]:
        """Forward pass of the visualizer.

        Takes an image and the prepared inputs from the `prepare` method and
        produces visualizations. Visualizations can be either::

            1. A single image (e.g. for classification, weight visualization).
            2. A tuple of two images, representing (labels, predictions) (e.g. for
              bounding boxes, keypoints).
            3. A tuple of an image and a list of images,
              representing (labels, multiple visualizations) (e.g. for segmentation,
              depth estimation).
            4. A list of images, representing unrelated visualizations.

        Args:
            label_canvas (Tensor): An image to draw the labels on.
            prediction_canvas (Tensor): An image to draw the predictions on.
            *args: Prepared inputs from the `prepare` method.

        Returns:
            Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]: Visualizations.

        Raises:
            IncompatibleException: If the inputs are not compatible with the module.
        """
        ...

    def run(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        inputs: Packet[Tensor],
        labels: Labels,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        self.validate(inputs, labels)
        return self(label_canvas, prediction_canvas, *self.prepare(inputs, labels))
