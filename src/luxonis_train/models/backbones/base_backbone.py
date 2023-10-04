import torch.nn as nn
from abc import ABC, abstractmethod
import warnings
from typing import Optional


class BaseBackbone(nn.Module, ABC):
    def __init__(self, **kwargs):
        """Base abstract backbone class from which all other heads are created"""
        super().__init__()

        if len(kwargs):
            warnings.warn(
                f"Following backbone parameters for `{self.get_name()}` not used: {kwargs}"
            )

    @abstractmethod
    def forward(self, x: any):
        """torch.nn.Module forward method"""
        pass

    def get_name(self, idx: Optional[int] = None):
        """Generate a string backbone name based on class name and passed index (if present)"""
        class_name = self.__class__.__name__
        if idx is not None:
            class_name += f"_{idx}"
        return class_name
