#
# Source: https://github.com/WangChyanhassth-2say/BlazePose_torch/tree/main
# License: MIT License
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from luxonis_train.models.modules import ConvModule, MobileBottleneck

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()
        self.use_pooling = stride == 2
        self.channel_pad = out_channels - in_channels

        if self.use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            padding = 0
        else:
            padding = 1

        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.use_pooling:
            conv_input = F.pad(x, [0, 1, 0, 1], "constant", 0)
            x = self.pool(x)
        else:
            conv_input = x

        conv_out = self.depth_conv(conv_input)
        conv_out = self.pointwise_conv(conv_out)

        if self.channel_pad > 0:
            x = F.pad(x, [0, 0, 0, 0, 0, self.channel_pad], "constant", 0)

        return self.relu(conv_out + x)

class BlazePose(nn.Module):
    def __init__(self, in_channels=3):
        super(BlazePose, self).__init__()
        self.in_channels = in_channels

        self.conv1 = ConvModule(in_channels, 16, 3, 2, 1, activation=nn.SiLU())

        self.conv2_b1 = MobileBottleneck(16, 16, 3, 1, 72, False, nn.ReLU6())
        self.conv3_b1 = MobileBottleneck(32, 32, 5, 1, 120, True, nn.ReLU6())
        self.conv4_b1 = MobileBottleneck(64, 64, 3, 1, 200, False, nn.SiLU())
        self.conv4_b2 = MobileBottleneck(64, 64, 3, 1, 184, False, nn.SiLU())
        self.conv5_b1 = MobileBottleneck(128, 128, 3, 1, 480, True, nn.SiLU())
        self.conv5_b2 = MobileBottleneck(128, 128, 3, 1, 672, True, nn.SiLU())
        self.conv6_b1 = MobileBottleneck(192, 192, 5, 1, 960, True, nn.SiLU())

        # blaze blocks
        self.conv2 = BlazeBlock(16, 16, stride=1)
        self.conv3 = BlazeBlock(16, 32, stride=2)
        self.conv4 = BlazeBlock(32, 64, stride=2)
        self.conv5 = BlazeBlock(64, 128, stride=2)
        self.conv6 = BlazeBlock(128, 192, stride=2)

        self.conv7_ = ConvModule(192, 32, 3, 1, 1, activation=nn.SiLU())
        self.conv8_ = ConvModule(128, 32, 3, 1, 1, activation=nn.SiLU())
        self.conv9_ = ConvModule(64, 32, 3, 1, 1, activation=nn.SiLU())
        
        # up sample layer
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._initialize_weights()

    def forward(self, x):
        outs = []
        # stem layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2_b1(x)
        
        # blazeblocks and mobilebottlenecks
        # naming differently for the skip connection
        y0 = self.conv3(x)
        y0 = self.conv3_b1(y0)
        outs.append(y0)

        y1 = self.conv4(y0)
        y1 = self.conv4_b1(y1)
        y1 = self.conv4_b2(y1)
        outs.append(y1)

        y2 = self.conv5(y1)
        y2 = self.conv5_b1(y2)
        y2 = self.conv5_b2(y2)
        outs.append(y2)

        y3 = self.conv6(y2)
        y3 = self.conv6_b1(y3)
        outs.append(y3)

        # heatmap branch
        x3 = self.conv7_(y3)
        x2 = self.conv8_(y2) + self.upsample2(x3)
        x1 = self.conv9_(y1) + self.upsample1(x2)
        x0 = y0 + self.upsample0(x1)
        outs.append(x0)

        return outs

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

if __name__ == '__main__':
    model = BlazePose()
    model.eval()
    # print(model)
    shapes = [224, 256, 384, 512]

    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        for out in outs:
            print(out.shape)