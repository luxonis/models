from abc import ABC, abstractmethod
import torch.nn as nn
import warnings
from torch import Tensor
from typing import Optional, Tuple, Dict, Any


class BaseLoss(nn.Module, ABC):
    def __init__(self, head_attributes: Dict[str, Any] = {}, **kwargs):
        """Base abstract loss class from which all other losses are created

        Args:
            head_attributes (Dict[str, Any], optional): Dictionary of all head
                attributes to which the loss is connected to. Defaults to {}.
        """
        super().__init__()

        self.head_attributes = head_attributes

        if len(kwargs):
            warnings.warn(
                f"Following loss parameters for `{self.get_name()}` not used: {kwargs}"
            )

    @abstractmethod
    def forward(
        self, preds: Any, target: Any, epoch: int, step: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """torch.nn.Module forward method

        Args:
            preds (Any): Processed model predictions
            target (Any): Target data
            epoch (int): Current training epoch
            step (int): Current training step

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: Output loss tensor, dict of other sub losses
                (empty dict if there aren't any)
        """
        pass

    def get_name(self, idx: Optional[int] = None):
        """Generate a string loss name based on class name and passed index (if present)"""
        class_name = self.__class__.__name__
        if idx is not None:
            class_name += f"_{idx}"
        return class_name
