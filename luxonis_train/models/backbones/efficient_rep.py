#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/efficientrep.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#

import torch.nn as nn
import torch

from luxonis_train.models.modules import RepVGGBlock, RepBlock, SimplifiedSPPF
from luxonis_train.utils.general import make_divisible

class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(self, in_channels=3, channels_list=None, num_repeats=None, depth_mul=0.33, width_mul=0.25):
        super(EfficientRep, self).__init__()

        if channels_list is None:
            raise ValueError("'channel_list' cannot be None")
        if num_repeats is None:
            raise ValueError("'num_repeats' cannot be None")

        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats]

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.blocks = nn.ModuleList()
        for i in range(4):
            curr_block = nn.Sequential(
                RepVGGBlock(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i+1],
                    kernel_size=3,
                    stride=2
                ),
                RepBlock(
                    in_channels=channels_list[i+1],
                    out_channels=channels_list[i+1],
                    n=num_repeats[i+1],
                )
            )
            if i == 3:
                curr_block.append(
                    SimplifiedSPPF(
                        in_channels=channels_list[i+1],
                        out_channels=channels_list[i+1],
                        kernel_size=5
                    )
                )
            
            self.blocks.append(curr_block)

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0:
                outputs.append(x)

        return outputs


if __name__ == "__main__":
    num_repeats = [1, 6, 12, 18, 6]
    depth_mul = 0.33
    
    channels_list =[64, 128, 256, 512, 1024]
    width_mul = 0.25

    model = EfficientRep(in_channels=3, channels_list=channels_list, num_repeats=num_repeats, depth_mul=depth_mul, width_mul=width_mul)
    model.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        for out in outs:
            print(out.shape)    
