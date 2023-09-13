#
# Soure: https://github.com/rwightman/gen-efficientnet-pytorch
# License: https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/LICENSE
#


import torch

from luxonis_train.models.backbones.base_backbone import BaseBackbone
from luxonis_train.utils.registry import BACKBONES


@BACKBONES.register_module()
class EfficientNet(BaseBackbone):
    def __init__(self, download_weights: bool = False, **kwargs):
        """EfficientNet backbone

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

    def forward(self, x):
        outs = []
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        for i, m in enumerate(self.backbone.blocks):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
