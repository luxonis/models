from abc import ABC, abstractmethod

from torch import Tensor, nn

from luxonis_train.utils.registry import MODULES, AutoRegisterMeta
from luxonis_train.utils.types import (
    ModulePacket,
    ShapePacket,
)


class LuxonisModule(
    nn.Module,
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
):
    _REGISTRY = MODULES
    """
    Abstract methods:
        init_module(**kwargs) -> None:
            initialize the module
        forward_pass(inputs: list[Tensor]) -> list[Tensor]:
            forward pass of the module
    """

    def __init__(self, input_shapes: list[ShapePacket], **kwargs):
        super().__init__(**kwargs)

        self.input_shapes = input_shapes
        self.loss = None

    @abstractmethod
    def validate(self, inputs: list[Tensor]) -> None:
        ...

    @abstractmethod
    def forward(self, inputs: list[ModulePacket]) -> ModulePacket:
        ...
