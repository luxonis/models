import torch 
import math
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.utils import draw_segmentation_masks

from luxonis_train.models.heads.base_heads import BaseSegmentationHead
from luxonis_train.utils.visualization import (
    torch_img_to_numpy,
    numpy_to_torch_img,
    seg_output_to_bool,
)


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class SpatialPath(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
        ch = 64
        self.conv_7x7 = ConvModule(c1, ch, 7, 2, 3)
        self.conv_3x3_1 = ConvModule(ch, ch, 3, 2, 1)
        self.conv_3x3_2 = ConvModule(ch, ch, 3, 2, 1)
        self.conv_1x1 = ConvModule(ch, c2, 1, 1, 0)

    def forward(self, x):           
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class ContextPath(nn.Module):
    def __init__(self, backbone=None) -> None:
        super().__init__()
        self.backbone = backbone
        # backbone or neck last two channels
        c3, c4 = 176, 344

        self.arm16 = AttentionRefinmentModule(c3, 128)
        self.arm32 = AttentionRefinmentModule(c4, 128)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(c4, 128, 1, 1, 0)
        )

        self.up16 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)

        self.refine16 = ConvModule(128, 128, 3, 1, 1)
        self.refine32 = ConvModule(128, 128, 3, 1, 1)


    def forward(self, x, y):
        down16, down32 = x, y                 # 4x256x64x128, 4x512x32x64

        arm_down16 = self.arm16(down16)                 # 4x128x64x128
        arm_down32 = self.arm32(down32)                 # 4x128x32x64

        global_down32 = self.global_context(down32)     # 4x128x1x1
        global_down32 = F.interpolate(global_down32, size=down32.size()[2:], mode='bilinear', align_corners=True)   # 4x128x32x64

        arm_down32 = arm_down32 + global_down32                         # 4x128x32x64
        arm_down32 = self.up32(arm_down32)                  # 4x128x64x128
        arm_down32 = self.refine32(arm_down32)              # 4x128x64x128

        arm_down16 = arm_down16 + arm_down32                            # 4x128x64x128
        arm_down16 = self.up16(arm_down16)                  # 4x128x128x256
        arm_down16 = self.refine16(arm_down16)              # 4x128x128x256      

        return arm_down16, arm_down32


class AttentionRefinmentModule(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
        self.conv_3x3 = ConvModule(c1, c2, 3, 1, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        return fm * fm_se


class FeatureFusionModule(nn.Module):
    def __init__(self, c1, c2, reduction=1) -> None:
        super().__init__()
        self.conv_1x1 = ConvModule(c1, c2, 1, 1, 0)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(c2 // reduction, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)
        return fm + fm * fm_se


class Head(nn.Module):
    def __init__(self, c1, n_classes, upscale_factor, is_aux=False) -> None:
        super().__init__()
        ch = 256 if is_aux else 64
        c2 = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x))
        return self.upscale(x)


class BiSeNetv1(BaseSegmentationHead):
    def __init__(self, n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list, attach_index: int = -1, backbone: str = 'ResNet-18', num_classes: int = 2, using_separation_loss=False, **kwargs) -> None:
        super().__init__(n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
            **kwargs)

        self.context_path = ContextPath()
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionModule(256, 256)

        self.output_head = Head(256, num_classes, upscale_factor=8, is_aux=False)
        self.context16_head = Head(128, num_classes, upscale_factor=8, is_aux=True)
        self.context32_head = Head(128, num_classes, upscale_factor=16, is_aux=True)

        self.using_separation_loss = using_separation_loss
        self.apply(self._init_weights)
    
    def postprocess_for_metric(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]
        if self.n_classes != 1:
            label = torch.argmax(label, dim=1)
        return output[0], label, None

    def postprocess_for_loss(self, output: torch.Tensor, label_dict: dict):
        label = label_dict[self.label_types[0]]

        return output, label
    def draw_output_to_img(self, img: torch.Tensor, output: torch.Tensor, idx: int):
        masks = seg_output_to_bool(output[0][idx])
        # NOTE: we have to push everything to cpu manually before draw_segmentation_masks (torchvision bug?)
        masks = masks.cpu()
        img = img.cpu()
        img = draw_segmentation_masks(img, masks, alpha=0.4)
        return img

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.context_path.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x, down16, down32):                                       # 4x3x1024x2048           
        spatial_out = self.spatial_path(x)                      # 4x128x128x256
        context16, context32 = self.context_path(down16, down32)             # 4x128x128x256, 4x128x64x128

        fm_fuse = self.ffm(spatial_out, context16)              # 4x256x128x256
        output = self.output_head(fm_fuse)                      # 4xn_classesx1024x2048

        context_out16 = self.context16_head(context16)      # 4xn_classesx1024x2048
        context_out32 = self.context32_head(context32)      # 4xn_classesx1024x2048
        if self.using_separation_loss:

            return [output, context32, context_out32]
        else:
            return [output, context_out16, context_out32]