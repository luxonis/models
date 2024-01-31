from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import Tensor
from torchmetrics import Metric

from luxonis_train.utils.registry import METRICS, AutoRegisterMeta
from luxonis_train.utils.types import Labels, LabelType, ModulePacket

TargetType = TypeVar("TargetType")
PredictionType = TypeVar("PredictionType")


class LuxonisMetric(
    Metric,
    ABC,
    Generic[PredictionType, TargetType],
    metaclass=AutoRegisterMeta,
    register=False,
):
    _REGISTRY = METRICS
    # TODO automatic/better naming
    name: str

    @abstractmethod
    def preprocess(
        self, outputs: ModulePacket, labels: Labels
    ) -> tuple[PredictionType, TargetType]:
        ...

    @abstractmethod
    def update(self, preds: PredictionType, target: TargetType) -> None:
        ...

    @staticmethod
    def _default_preprocess(
        outputs: ModulePacket, labels: Labels
    ) -> tuple[Tensor, Tensor]:
        output_keys = list(outputs.keys())
        if "features" in output_keys:
            output_keys.remove("features")
        if len(output_keys) != 1:
            raise ValueError("The metric cannot be created automatically")
        key = output_keys[0]
        output = outputs[key][0]
        label = labels[LabelType(key)]

        if len(output) != len(label):
            raise ValueError("The metric cannot be created automatically")
        return output, label
