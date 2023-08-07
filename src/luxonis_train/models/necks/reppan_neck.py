#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#


import torch
import torch.nn as nn

from luxonis_train.models.modules import RepBlock, ConvModule
from luxonis_train.utils.general import make_divisible


class RepPANNeck(nn.Module):
    def __init__(
        self,
        prev_out_shapes: list,
        channels_list: list,
        num_repeats: list,
        depth_mul: float = 0.33,
        width_mul: float = 0.25,
        is_4head: bool = False,
        **kwargs
    ):
        """RepPANNeck normally used with YoloV6 model. It has the balance of feature fusion ability and hardware efficiency.

        Args:
            prev_out_shapes (list): List of shapes of previous outputs
            channels_list (list): List of number of channels for each block.
            num_repeats (list): List of number of repeats of RepBlock
            depth_mul (float, optional): Depth multiplier. Defaults to 0.33.
            width_mul (float, optional): Width multiplier. Defaults to 0.25.
            is_4head (bool, optional): Either build 4 headed architecture or 3 headed one \
                (**Important: Should be same also on backbone and head**). Defaults to False.
        """
        super().__init__()

        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]

        self.is_4head = is_4head
        prev_out_start_idx = 1 if self.is_4head else 0

        self.Rep_p4 = RepBlock(
            in_channels=prev_out_shapes[prev_out_start_idx + 1][1] + channels_list[0],
            out_channels=channels_list[0],
            n=num_repeats[0],
        )

        self.Rep_p3 = RepBlock(
            in_channels=prev_out_shapes[prev_out_start_idx][1] + channels_list[1],
            out_channels=channels_list[1],
            n=num_repeats[1],
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[1] + channels_list[2],
            out_channels=channels_list[3],
            n=num_repeats[2],
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[0] + channels_list[4],
            out_channels=channels_list[5],
            n=num_repeats[3],
        )

        self.reduce_layer0 = ConvModule(
            in_channels=prev_out_shapes[prev_out_start_idx + 2][1],
            out_channels=channels_list[0],
            kernel_size=1,
            stride=1,
        )

        self.upsample0 = torch.nn.ConvTranspose2d(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=2,
            stride=2,
            bias=True,
        )

        self.reduce_layer1 = ConvModule(
            in_channels=channels_list[0],
            out_channels=channels_list[1],
            kernel_size=1,
            stride=1,
        )

        self.upsample1 = torch.nn.ConvTranspose2d(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=2,
            stride=2,
            bias=True,
        )

        self.downsample2 = ConvModule(
            in_channels=channels_list[1],
            out_channels=channels_list[2],
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )

        self.downsample1 = ConvModule(
            in_channels=channels_list[3],
            out_channels=channels_list[4],
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )

        if self.is_4head:
            self.reduce_layer2 = ConvModule(
                in_channels=channels_list[1],
                out_channels=channels_list[1] // 2,
                kernel_size=1,
                stride=1,
            )
            self.upsample2 = torch.nn.ConvTranspose2d(
                in_channels=channels_list[1] // 2,
                out_channels=channels_list[1] // 2,
                kernel_size=2,
                stride=2,
                bias=True,
            )
            self.Rep_p2 = RepBlock(
                in_channels=prev_out_shapes[prev_out_start_idx - 1][1]
                + channels_list[1] // 2,
                out_channels=channels_list[1] // 2,
                n=num_repeats[1],
            )
            self.downsample3 = ConvModule(
                in_channels=channels_list[1] // 2,
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2,
                padding=3 // 2,
            )
            self.Rep_n2 = RepBlock(
                in_channels=channels_list[1] + channels_list[1] // 2,
                out_channels=channels_list[1],
                n=num_repeats[1],
            )

    def forward(self, x):
        if self.is_4head:
            x3, x2, x1, x0 = x
        else:
            x2, x1, x0 = x

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], dim=1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], dim=1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        if self.is_4head:
            fpn_out2 = self.reduce_layer2(pan_out2)
            upsample_feat2 = self.upsample2(fpn_out2)
            f_concat_layer2 = torch.cat([upsample_feat2, x3], dim=1)
            pan_out3 = self.Rep_p2(f_concat_layer2)

            down_feat2 = self.downsample3(pan_out3)
            p_concat_layer0 = torch.cat([down_feat2, fpn_out2], dim=1)
            pan_out2 = self.Rep_n2(p_concat_layer0)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], dim=1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], dim=1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        if self.is_4head:
            outputs = [pan_out3, pan_out2, pan_out1, pan_out0]
        else:
            outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


if __name__ == "__main__":
    # test together with EfficientRep backbone
    from luxonis_train.models.backbones import EfficientRep
    from luxonis_train.utils.general import dummy_input_run

    num_repeats_backbone = [1, 6, 12, 18, 6]
    num_repeats_neck = [12, 12, 12, 12]
    depth_mul = 0.33

    channels_list_backbone = [64, 128, 256, 512, 1024]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    width_mul = 0.25

    backbone = EfficientRep(
        in_channels=3,
        channels_list=channels_list_backbone,
        num_repeats=num_repeats_backbone,
        depth_mul=depth_mul,
        width_mul=width_mul,
        is_4head=False,
    )
    backbone_out_shapes = dummy_input_run(backbone, [1, 3, 224, 224])
    backbone.eval()

    neck = RepPANNeck(
        prev_out_shape=backbone_out_shapes,
        channels_list=channels_list_neck,
        num_repeats=num_repeats_neck,
        depth_mul=depth_mul,
        width_mul=width_mul,
        is_4head=False,
    )
    neck.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        outs = neck(outs)
        for out in outs:
            print(out.shape)
