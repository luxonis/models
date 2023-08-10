import torch.nn as nn
from abc import ABC, abstractmethod


class BaseBackbone(nn.Module, ABC):
    def __init__(self):
        """Base abstract backbone class from which all other heads are created"""
        super().__init__()

    @abstractmethod
    def forward(self, x: any):
        """torch.nn.Module forward method"""
        pass
