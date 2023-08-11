#
# Soure: https://github.com/rwightman/gen-efficientnet-pytorch
# License: https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/LICENSE
#


import torch
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, download_weights: bool = False):
        """EfficientNet backbone

        Args:
            download_weights (bool, optional): If True download weights from imagenet. Defaults to False.
        """
        super().__init__()
        efficientnet_lite0_model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "efficientnet_lite0",
            pretrained=download_weights,
        )
        self.out_indices = [1, 2, 4, 6]
        self.backbone = efficientnet_lite0_model

    def forward(self, X):
        outs = []
        X = self.backbone.conv_stem(X)
        X = self.backbone.bn1(X)
        X = self.backbone.act1(X)
        for i, m in enumerate(self.backbone.blocks):
            X = m(X)
            if i in self.out_indices:
                outs.append(X)
        return outs


if __name__ == "__main__":
    model = EfficientNet()
    model.eval()

    shapes = [224, 256, 384, 512]

    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        if isinstance(outs, list):
            for out in outs:
                print(out.shape)
        else:
            print(outs.shape)
