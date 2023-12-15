from abc import abstractmethod

from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.utils.registry import METRICS
from luxonis_train.utils.types import Labels, Packet

Ts = TypeVarTuple("Ts")


class BaseMetric(
    BaseAttachedModule[Unpack[Ts]],
    Metric,
    register=False,
    registry=METRICS,
):
    """A base class for all metrics.

    This class defines the basic interface for all metrics. It utilizes automatic
    registration of defined subclasses to a `METRICS` registry.
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
            Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]: The computed metric. Can be one of::

                1. A single `Tensor`.
                2. A tuple of a `Tensor` and a dictionary of submetrics.
                3. A dictionary of submetrics. If this is the case, then the metric
                  cannot be used as the main metric of the model.
        """
        ...

    def run_update(self, outputs: Packet[Tensor], labels: Labels) -> None:
        """Calls the metric's update method.

        Validates and prepares the inputs, then calls the metric's update method.

        Args:
            outputs (Packet[Tensor]): The outputs of the model.
            labels (Labels): The labels of the model.

        Raises:
            IncompatibleException: If the inputs are not compatible with the module.
        """
        self.validate(outputs, labels)
        self.update(*self.prepare(outputs, labels))
