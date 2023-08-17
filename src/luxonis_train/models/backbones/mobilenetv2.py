#
# Soure: https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html
# License: https://github.com/pytorch/pytorch/blob/master/LICENSE
#


import torchvision

from luxonis_train.models.backbones.base_backbone import BaseBackbone


class MobileNetV2(BaseBackbone):
    def __init__(self, download_weights: bool = False, **kwargs):
        """MobileNetV2 backbone

        Args:
            download_weights (bool, optional): If True download weights from imagenet. Defaults to False.
        """
        super().__init__(**kwargs)

        mobilenet_v2 = torchvision.models.mobilenet_v2(
            weights="DEFAULT" if download_weights else None
        )
        self.out_indices = [3, 6, 13, 17]
        self.channels = [24, 32, 96, 320]
        self.backbone = mobilenet_v2

    def forward(self, X):
        outs = []
        for i, m in enumerate(self.backbone.features):
            X = m(X)
            if i in self.out_indices:
                outs.append(X)
        return outs
