from abc import abstractmethod

from torch import Tensor
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import LuxonisAttachedModule
from luxonis_train.utils.registry import VISUALIZERS
from luxonis_train.utils.types import Labels, Packet

Ts = TypeVarTuple("Ts")


class LuxonisVisualizer(
    LuxonisAttachedModule[Unpack[Ts]],
    register=False,
    registry=VISUALIZERS,
):
    """A base class for all visualizers.

    This class defines the basic interface for all visualizers. It utilizes automatic
    registration of defined subclasses to a `VISUALIZERS` registry.

    Metaclass Args:
        register (bool): Determines whether or not to register this class.
          Should be set to False in abstract classes to prevent them
          from being registered.
        registry (Registry): The registry to which the subclasses should be added.
          For most times should not be specified in concrete classes.

    Interface:
        prepare(outputs: Packet[Tensor], labels: Labels
        ) -> tuple[Unpack[Ts]]:
          Prepares the outputs and labels before passing them to the forward method.
          Should allow for the following call: `forward(*prepare(outputs, labels))`.

        @abstractmethod
        forward(label_canvas: Tensor, prediction_canvas: Tensor, *args: Unpack[Ts]
        ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]:
            Forward pass of the visualizer. Takes an image and the prepared inputs from
            `prepare` method and produces visualizations. Visualizations can be either:
                1. A single image (e.g. for classification, weight visualization).
                2. A tuple of two images, representing (labels, predictions) (e.g. for
                  bounding boxes, keypoints).
                3. A tuple of an image and a list of images,
                  representing (labels, multiple visualizations) (e.g. for segmentation,
                  depth estimation).
                4. A list of images, representing unrelated visualizations.
    """

    @abstractmethod
    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        idx: int,
        *args: Unpack[Ts],
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]:
        """Forward pass of the visualizer.

        Takes an image and the prepared inputs from the `prepare` method and
        produces visualizations. Visualizations can be either:
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
            idx (int): Index of the image in the batch.
            *args: Prepared inputs from the `prepare` method.

        Returns:
            Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]: Visualizations.
        """
        ...

    def __call__(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        inputs: Packet[Tensor],
        labels: Labels,
        idx: int = 0,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        self.validate(inputs, labels)
        return super().__call__(
            label_canvas, prediction_canvas, idx, *self.prepare(inputs, labels)
        )
