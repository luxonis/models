from abc import abstractmethod

from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import LuxonisAttachedModule
from luxonis_train.utils.registry import METRICS

Ts = TypeVarTuple("Ts")


class LuxonisMetric(
    LuxonisAttachedModule[Unpack[Ts]],
    Metric,
    register=False,
    registry=METRICS,
):
    """A base class for all metrics.

    This class defines the basic interface for all metrics. It utilizes automatic
    registration of defined subclasses to a `METRICS` registry.

    Metaclass Args:
        register (bool): Determines whether or not to register this class.
          Should be set to False in abstract classes to prevent them
          from being registered.
        registry (dict): The registry to which the subclasses should be added.
          For most times should not be specified in concrete classes.

    Interface:
        prepare(outputs: Packet[Tensor], labels: Labels
        ) -> tuple[Unpack[Ts]]:
            Prepares the outputs and labels before passing them to the forward method.
            Should allow for the following call: `update(*prepare(outputs, labels))`.

        @abstractmethod
        forward(*args: Unpack[Ts]) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
            Forward pass of the loss function. Produces either a single `Tensor`, or
            a tupel of a `Tensor` (main loss) and sublosses. Sublosses are used for
            logging purposes and are not used in backpropagation.

        @abstractmethod
        update(*args: Unpack[Ts]) -> None:
            Updates the inner state of the metric.

        @abstractmethod
        compute() -> Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
            Computes the final value(s) of the metric. Should return either a single
            `Tensor`, a tuple of a `Tensor` and a dictionary of submetrics, or only a
            dictionary of submetrics. If the latest case is true, then the metric
            cannot be used as the main metric of the model.
    """

    @abstractmethod
    def update(self, *args: Unpack[Ts]) -> None:
        """Updates the inner state of the metric.

        Args:
            *args: Prepared inputs from the `prepare` method.
        """
        ...

    @abstractmethod
    def compute(self) -> Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
        """Computes the metric.

        Returns:
            Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]: The computed metric. Can be one of:
                1. A single `Tensor`.
                2. A tuple of a `Tensor` and a dictionary of submetrics.
                3. A dictionary of submetrics. If this is the case, then the metric
                  cannot be used as the main metric of the model.
        """
        ...
