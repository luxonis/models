from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import Tensor, nn

from luxonis_train.utils.registry import LOSSES, AutoRegisterMeta
from luxonis_train.utils.types import (
    Kwargs,
    Labels,
    ModulePacket,
)

TargetType = TypeVar("TargetType")
PredictionType = TypeVar("PredictionType")


class LuxonisLoss(
    nn.Module,
    ABC,
    Generic[PredictionType, TargetType],
    metaclass=AutoRegisterMeta,
    register=False,
):
    _REGISTRY = LOSSES

    # TODO: maybe change to reference to module
    def __init__(self, module_attributes: Kwargs | None = None, **kwargs):
        super().__init__(**kwargs)
        self.module_attributes = module_attributes or {}

    @abstractmethod
    def compute_loss(
        self, prediction: PredictionType, target: TargetType
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pass

    @abstractmethod
    def validate(self, outputs: ModulePacket, labels: Labels) -> None:
        pass

    # TODO: IncompatibleException
    @abstractmethod
    def preprocess(
        self, outputs: ModulePacket, labels: Labels
    ) -> tuple[PredictionType, TargetType]:
        ...

    def forward(
        self, outputs: ModulePacket, labels: Labels
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.validate(outputs, labels)
        predictions, target = self.preprocess(outputs, labels)
        return self.compute_loss(predictions, target)
