"""ResNet18 backbone.

Source: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
License (PyTorch): https://github.com/pytorch/pytorch/blob/master/LICENSE
"""


import torchvision
from torch import Tensor

from .luxonis_node import LuxonisNode


class ResNet18(LuxonisNode[Tensor, list[Tensor]]):
    def __init__(
        self,
        channels_list: list[int] | None = None,
        download_weights: bool = False,
        **kwargs,
    ):
        """Implementation of the ResNet18 backbone.

        TODO: add more info

        Args:
            download_weights (bool, optional): If True download weights from imagenet.
              Defaults to False.
        """
        super().__init__(attach_index=-1, **kwargs)

        self.backbone = torchvision.models.resnet18(
            weights="DEFAULT" if download_weights else None
        )
        self.channels_list = channels_list or [64, 128, 256, 512]

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        outs.append(x)
        x = self.backbone.layer2(x)
        outs.append(x)
        x = self.backbone.layer3(x)
        outs.append(x)
        x = self.backbone.layer4(x)
        outs.append(x)

        return outs
