#
# Soure: https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html
# License: https://github.com/pytorch/pytorch/blob/master/LICENSE
#


import torch
import torch.nn as nn
import torchvision


class MobileNetV2(nn.Module):

    def __init__(self, pretrained=False):
        super(MobileNetV2, self).__init__()
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights="DEFAULT" if pretrained else None)
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


if __name__ == '__main__':

    model = MobileNetV2(pretrained=True)
    model.eval()
    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        for out in outs:
            print(out.shape)
