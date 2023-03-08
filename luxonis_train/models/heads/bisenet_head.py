#
# Source: https://github.com/taveraantonio/BiseNetv1
#

import torch
import torch.nn as nn

from luxonis_train.models.modules import ConvModule
from luxonis_train.utils.head_type import *

class BiSeNetHead(nn.Module):
    def __init__(self, prev_out_shape, n_classes, c1=256, upscale_factor=8, is_aux=False, **kwargs) -> None:
        super(BiSeNetHead, self).__init__()

        self.n_classes = n_classes
        self.type = SemanticSegmentation()
        self.original_in_shape = kwargs["original_in_shape"]
        self.prev_out_shape = prev_out_shape

        ch = 256 if is_aux else 64
        c2 = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x))
        return self.upscale(x)

if __name__ == "__main__":

    from luxonis_train.models.backbones import ContextSpatial

    backbone = ContextSpatial()
    backbone.eval()

    head = BiSeNetHead(n_classes=2)
    head.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        outs = head(outs)
        for out in outs:
            print(out.shape)
