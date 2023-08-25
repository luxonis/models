from abc import ABC, abstractmethod
import torch.nn as nn
import warnings
from typing import Optional


class BaseLoss(nn.Module, ABC):
    def __init__(self, head_attributes: dict, **kwargs):
        """Base abstract loss class from which all other losses are created

        Args:
            head_attributes (dict): Dictionary of all head attributes to which the loss is connected to
        """
        super().__init__()

        self.head_attributes = head_attributes

        if len(kwargs):
            warnings.warn(
                f"Following loss parameters for `{self.get_name()}` not used: {kwargs}"
            )

    @abstractmethod
    def forward(self, outputs: any, targets: any, epoch: int, step: int):
        """torch.nn.Module forward method"""
        pass

    def get_name(self, idx: Optional[int] = None):
        """Generate a string loss name based on class name and passed index (if present)"""
        class_name = self.__class__.__name__
        if idx is not None:
            class_name += f"_{idx}"
        return class_name
