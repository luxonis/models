import torch.nn as nn
from abc import ABC, abstractmethod


class BaseNeck(nn.Module, ABC):
    def __init__(self, input_channels_shapes: list):
        """Base abstract class from which all other necks are created

        Args:
            input_channels_shapes (list): List of output shapes from previous module.
        """
        super().__init__()

        self.input_channels_shapes = input_channels_shapes

    @abstractmethod
    def forward(self, x: any):
        """torch.nn.Module forward method"""
        pass
