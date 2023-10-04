#
# Soure: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
# License: https://github.com/pytorch/pytorch/blob/master/LICENSE
#


import torchvision

from luxonis_train.models.backbones.base_backbone import BaseBackbone


class ResNet18(BaseBackbone):
    def __init__(self, download_weights: bool = False, **kwargs):
        """ResNet18 backbone

        Args:
            download_weights (bool, optional): If True download weights from imagenet. Defaults to False.
        """
        super().__init__(**kwargs)

        resnet18 = torchvision.models.resnet18(
            weights="DEFAULT" if download_weights else None
        )
        self.channels = [64, 128, 256, 512]
        self.backbone = resnet18

    def forward(self, x):
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
