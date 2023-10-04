import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        """Hard-Sigmoid (approximated sigmoid) activation function from 'Searching for MobileNetV3,'
        https://arxiv.org/abs/1905.02244."""
        self.relu = nn.ReLU6(True)

    def forward(self, x: torch.Tensor):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        """H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244."""
        self.sigmoid = HSigmoid()

    def forward(self, x: torch.Tensor):
        return x * self.sigmoid(x)
