from abc import abstractmethod

from torch import Tensor
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.utils.registry import LOSSES
from luxonis_train.utils.types import (
    Labels,
    Packet,
)

Ts = TypeVarTuple("Ts")


class BaseLoss(
    BaseAttachedModule[Unpack[Ts]],
    register=False,
    registry=LOSSES,
):
    """A base class for all loss functions.

    This class defines the basic interface for all loss functions. It utilizes automatic
    registration of defined subclasses to a `LOSSES` registry.

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
        forward(*args: Unpack[Ts]) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
          Forward pass of the loss function. Produces either a single `Tensor`, or
          a tupel of a `Tensor` (main loss) and sublosses. Sublosses are used for
          logging purposes and are not used in backpropagation.
    """

    @abstractmethod
    def forward(self, *args: Unpack[Ts]) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Forward pass of the loss function.

        Args:
            *args: Prepared inputs from the `prepare` method.

        Returns:
            tuple[Tensor, dict[str, Tensor]] | Tensor: The main loss and optional
              a dictionary of sublosses (for logging).
              Only the main loss is used for backpropagation.
        """
        ...

    def run(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Calls the loss function.

        Validates and prepares the inputs, then calls the loss function.

        Args:
            inputs (Packet[Tensor]): Outputs from the node.
            labels (Labels): Labels from the dataset.

        Returns:
            Tensor | tuple[Tensor, dict[str, Tensor]]: The main loss and optional
              a dictionary of sublosses (for logging).
              Only the main loss is used for backpropagation.

        Raises:
            IncompatibleException: If the inputs are not compatible with the module.
        """
        self.validate(inputs, labels)
        return self(*self.prepare(inputs, labels))
