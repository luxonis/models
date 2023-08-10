#
# Source: https://github.com/taveraantonio/BiseNetv1
#


import torch
import torch.nn as nn
from torch.nn import functional as F

from luxonis_train.models.backbones import *
from luxonis_train.models.backbones.base_backbone import BaseBackbone
from luxonis_train.models.modules import ConvModule


class ContextSpatial(BaseBackbone):
    def __init__(self, context_backbone: str = "MobileNetV2", **kwargs):
        """Context spatial backbone

        Args:
            context_backbone (str, optional): Backbone used. Defaults to 'MobileNetV2'.
        """
        super().__init__()

        self.context_path = ContextPath(eval(context_backbone)())
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionModule(256, 256)

    def forward(self, x):
        spatial_out = self.spatial_path(x)
        context16, context32 = self.context_path(x)
        fm_fuse = self.ffm(spatial_out, context16)
        outs = [fm_fuse]
        return outs


class SpatialPath(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        intermediate_channels = 64
        self.conv_7x7 = ConvModule(in_channels, intermediate_channels, 7, 2, 3)
        self.conv_3x3_1 = ConvModule(
            intermediate_channels, intermediate_channels, 3, 2, 1
        )
        self.conv_3x3_2 = ConvModule(
            intermediate_channels, intermediate_channels, 3, 2, 1
        )
        self.conv_1x1 = ConvModule(intermediate_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class ContextPath(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        c3, c4 = self.backbone.channels[-2:]

        self.arm16 = AttentionRefinmentModule(c3, 128)
        self.arm32 = AttentionRefinmentModule(c4, 128)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), ConvModule(c4, 128, 1, 1, 0)
        )

        self.up16 = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)

        self.refine16 = ConvModule(128, 128, 3, 1, 1)
        self.refine32 = ConvModule(128, 128, 3, 1, 1)

    def forward(self, x):
        _, _, down16, down32 = self.backbone(x)  # 4x256x64x128, 4x512x32x64

        arm_down16 = self.arm16(down16)  # 4x128x64x128
        arm_down32 = self.arm32(down32)  # 4x128x32x64

        global_down32 = self.global_context(down32)  # 4x128x1x1
        global_down32 = F.interpolate(
            global_down32, size=down32.size()[2:], mode="bilinear", align_corners=True
        )  # 4x128x32x64

        arm_down32 = arm_down32 + global_down32  # 4x128x32x64
        arm_down32 = self.up32(arm_down32)  # 4x128x64x128
        arm_down32 = self.refine32(arm_down32)  # 4x128x64x128

        arm_down16 = arm_down16 + arm_down32  # 4x128x64x128
        arm_down16 = self.up16(arm_down16)  # 4x128x128x256
        arm_down16 = self.refine16(arm_down16)  # 4x128x128x256

        return arm_down16, arm_down32


class AttentionRefinmentModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_3x3 = ConvModule(in_channels, out_channels, 3, 1, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        return fm * fm_se


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1):
        super().__init__()
        self.conv_1x1 = ConvModule(in_channels, out_channels, 1, 1, 0)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)
        return fm + fm * fm_se
