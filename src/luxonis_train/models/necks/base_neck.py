import torch.nn as nn
from abc import ABC, abstractmethod
import warnings
from typing import Optional


class BaseNeck(nn.Module, ABC):
    def __init__(self, input_channels_shapes: list, attach_index: int = -1, **kwargs):
        """Base abstract class from which all other necks are created

        Args:
            input_channels_shapes (list): List of output shapes from previous module.
            attach_index (int, optional): Index of previous output that the neck attaches to. Defaults to -1.
        """
        super().__init__()

        self._validate_attach_index(attach_index, len(input_channels_shapes))

        self.input_channels_shapes = input_channels_shapes
        self.attach_index = attach_index

        if len(kwargs):
            warnings.warn(
                f"Following neck parameters for `{self.get_name()}` not used: {kwargs}"
            )

    @abstractmethod
    def forward(self, x: any):
        """torch.nn.Module forward method"""
        pass

    def get_name(self, idx: Optional[int] = None):
        """Generate a string neck name based on class name and passed index (if present)"""
        class_name = self.__class__.__name__
        if idx is not None:
            class_name += f"_{idx}"
        return class_name

    def _validate_attach_index(self, index: int, inputs_len: int):
        """Validates attach index based on length of inputs

        Args:
            index (int): Attach index
            inputs_len (int): Length of inputs

        Raises:
            ValueError: Specified attach index out of range
        """
        if (index < 0 and abs(index) > inputs_len) or (
            index >= 0 and index >= inputs_len
        ):
            raise ValueError(
                f"Specified attach index for {self.get_name()} out of range."
            )
