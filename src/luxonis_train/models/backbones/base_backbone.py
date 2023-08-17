import torch.nn as nn
from abc import ABC, abstractmethod
import warnings


class BaseBackbone(nn.Module, ABC):
    def __init__(self, **kwargs):
        """Base abstract backbone class from which all other heads are created"""
        super().__init__()

        if len(kwargs):
            warnings.warn(f"Following backbone parameters not used: {kwargs}")

    @abstractmethod
    def forward(self, x: any):
        """torch.nn.Module forward method"""
        pass
