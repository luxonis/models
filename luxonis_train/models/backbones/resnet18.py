#
# Soure: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
# License: https://github.com/pytorch/pytorch/blob/master/LICENSE
#


import torch
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()

        resnet18 = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
        self.channels = [64, 128, 256, 512]
        self.backbone = resnet18

    def forward(self, X):
        outs = []
        X = self.backbone.conv1(X)
        X = self.backbone.bn1(X)
        X = self.backbone.relu(X)
        X = self.backbone.maxpool(X)

        X = self.backbone.layer1(X)
        outs.append(X)
        X = self.backbone.layer2(X)
        outs.append(X)
        X = self.backbone.layer3(X)
        outs.append(X)
        X = self.backbone.layer4(X)
        outs.append(X)

        return outs

if __name__ == '__main__':

    model = ResNet18()
    model.eval()

    shapes = [224, 256, 384, 512]

    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        for out in outs:
            print(out.shape)