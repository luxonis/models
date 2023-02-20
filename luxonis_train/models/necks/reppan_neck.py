#
# Soure: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#


import torch
import torch.nn as nn

from luxonis_train.models.modules import RepVGGBlock, RepBlock, ConvModule
from luxonis_train.utils.general import make_divisible

# _QUANT=False
class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(self, prev_out_shape, channels_list=None, num_repeats=None, depth_mul=0.33, width_mul=0.25, block=RepVGGBlock):
        super(RepPANNeck, self).__init__()

        assert channels_list, "channel_list can't be None"
        assert num_repeats, "num_repeats can't be None"

        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats]

        self.Rep_p4 = RepBlock(
            in_channels=prev_out_shape[1][1] + channels_list[0],
            out_channels=channels_list[0],
            n=num_repeats[0],
            block=block
        )

        self.Rep_p3 = RepBlock(
            in_channels=prev_out_shape[0][1] + channels_list[1],
            out_channels=channels_list[1],
            n=num_repeats[1],
            block=block
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[1] + channels_list[2],
            out_channels=channels_list[3],
            n=num_repeats[2],
            block=block
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[0] + channels_list[4],
            out_channels=channels_list[5],
            n=num_repeats[3],
            block=block
        )

        self.reduce_layer0 = ConvModule(
            in_channels=prev_out_shape[2][1],
            out_channels=channels_list[0],
            kernel_size=1,
            stride=1
        )

        self.upsample0 = torch.nn.ConvTranspose2d(
            in_channels=channels_list[0],
            out_channels=channels_list[0],
            kernel_size=2,
            stride=2,
            bias=True
        )

        self.reduce_layer1 = ConvModule(
            in_channels=channels_list[0],
            out_channels=channels_list[1],
            kernel_size=1,
            stride=1
        )

        self.upsample1 = torch.nn.ConvTranspose2d(
            in_channels=channels_list[1],
            out_channels=channels_list[1],
            kernel_size=2,
            stride=2,
            bias=True
        )

        self.downsample2 = ConvModule(
            in_channels=channels_list[1],
            out_channels=channels_list[2],
            kernel_size=3,
            stride=2,
            padding=3 // 2
        )

        self.downsample1 = ConvModule(
            in_channels=channels_list[3],
            out_channels=channels_list[4],
            kernel_size=3,
            stride=2,
            padding=3 // 2
        )

    def upsample_enable_quant(self, num_bits, calib_method):
        print("Insert fakequant after upsample")
        # Insert fakequant after upsample op to build TensorRT engine
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor
        conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)
        self.upsample_feat0_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        self.upsample_feat1_quant = quant_nn.TensorQuantizer(conv2d_input_default_desc)
        # global _QUANT
        self._QUANT = True

    def forward(self, x):
        x2, x1, x0 = x

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat0 = self.upsample_feat0_quant(upsample_feat0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        if hasattr(self, '_QUANT') and self._QUANT is True:
            upsample_feat1 = self.upsample_feat1_quant(upsample_feat1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

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

    backbone = EfficientRep(in_channels=3, channels_list=channels_list_backbone, 
        num_repeats=num_repeats_backbone, depth_mul=depth_mul, width_mul=width_mul)
    for module in backbone.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    backbone_out_shapes = dummy_input_run(backbone, [1,3,224,224])
    backbone.eval()

    neck = RepPANNeck(prev_out_shape=backbone_out_shapes, channels_list=channels_list_neck, 
        num_repeats=num_repeats_neck, depth_mul=depth_mul, width_mul=width_mul)
    neck.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        outs = neck(outs)
        for out in outs:
            print(out.shape)