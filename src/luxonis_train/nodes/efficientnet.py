"""Implementation of the EfficientNet backbone.

Source: `https://github.com/rwightman/gen-efficientnet-pytorch`
License (Apache 2.0): `https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/LICENSE`
"""

import torch
from torch import Tensor

from .luxonis_node import LuxonisNode


class EfficientNet(LuxonisNode[Tensor, list[Tensor]]):
    """EfficientNet backbone.

    TODO: DOCS
    """

    def __init__(self, download_weights: bool = False, **kwargs):
        """EfficientNet backbone.

        Args:
            download_weights (bool, optional): If True download weights from imagenet. Defaults to False.
        """
        super().__init__(**kwargs)

        efficientnet_lite0_model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "efficientnet_lite0",
            pretrained=download_weights,
        )
        self.out_indices = [1, 2, 4, 6]
        self.backbone = efficientnet_lite0_model

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        for i, m in enumerate(self.backbone.blocks):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
