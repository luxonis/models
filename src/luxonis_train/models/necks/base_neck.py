import torch.nn as nn
from abc import ABC, abstractmethod


class BaseNeck(nn.Module, ABC):
    def __init__(self, prev_out_shapes: list):
        """Base abstract class from which all other necks are created

        Args:
            prev_out_shapes (list): List of shapes of previous outputs
        """
        super().__init__()

        self.prev_out_shapes = prev_out_shapes

    @abstractmethod
    def forward(self, x: any):
        """torch.nn.Module forward method"""
        pass
