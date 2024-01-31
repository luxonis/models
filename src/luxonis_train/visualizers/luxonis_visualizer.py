from abc import ABC, abstractmethod

from torch import Tensor, nn

from luxonis_train.utils.registry import VISUALIZERS, AutoRegisterMeta
from luxonis_train.utils.types import Labels, ModulePacket


class LuxonisVisualizer(
    nn.Module,
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
):
    _REGISTRY = VISUALIZERS

    @abstractmethod
    def forward(
        self, img: Tensor, output: ModulePacket, label: Labels, idx: int = 0
    ) -> Tensor:
        ...
