#
# Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py
# License: https://github.com/meituan/YOLOv6/blob/main/LICENSE
#


import torch
import torch.nn as nn

from .effide_head import EffiDeHead
from luxonis_train.utils.head_type import ObjectDetection

class YoloV6Head(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, prev_out_shape, n_classes, num_layers=3, **kwargs):
        super(YoloV6Head, self).__init__()
        self.n_classes = n_classes  # number of classes
        self.type = ObjectDetection()
        self.original_in_shape = kwargs["original_in_shape"]
        self.prev_out_shape = prev_out_shape

        self.no = n_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers

        self.prior_prob = 1e-2

        self.n_anchors = 1
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.grid = [torch.zeros(1)] * num_layers
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        self.head = nn.ModuleList()
        for i in range(self.nl):
            curr_head = EffiDeHead(
                prev_out_shape=[prev_out_shape[i]],
                original_in_shape=self.original_in_shape,
                n_classes=self.n_classes,
                n_anchors=self.n_anchors
            )
            self.head.append(curr_head)

    def forward(self, x):
        cls_score_list = []
        reg_distri_list = []
        
        for i, module in enumerate(self.head):
            out_x, out_cls, out_reg = module([x[i]])
            x[i] = out_x
            out_cls = torch.sigmoid(out_cls)
            cls_score_list.append(out_cls.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(out_reg.flatten(2).permute((0, 2, 1)))

        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)

        return [x, cls_score_list, reg_distri_list]

    def to_deploy(self):
        # change definition of forward()
        def deploy_forward(x):
            outputs = []
            for i, module in enumerate(self.head):
                out_x, out_cls, out_reg = module([x[i]])
                out_cls = torch.sigmoid(out_cls)
                conf, _ = out_cls.max(1, keepdim=True)
                output = torch.cat([out_reg, conf, out_cls], axis=1)
                outputs.append(output)
            return outputs

        self.forward = deploy_forward


if __name__ == "__main__":
    # test yolov6-n config
    from luxonis_train.models.backbones import EfficientRep
    from luxonis_train.models.necks import RepPANNeck
    from luxonis_train.utils.general import dummy_input_run

    num_repeats_backbone = [1, 6, 12, 18, 6]
    num_repeats_neck = [12, 12, 12, 12]
    depth_mul = 0.33
    
    channels_list_backbone = [64, 128, 256, 512, 1024]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    width_mul = 0.25

    backbone = EfficientRep(in_channels=3, channels_list=channels_list_backbone, num_repeats=num_repeats_backbone,
        depth_mul=depth_mul, width_mul=width_mul)
    for module in backbone.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    backbone_out_shapes = dummy_input_run(backbone, [1,3,224,224])
    backbone.eval()

    neck = RepPANNeck(prev_out_shape=backbone_out_shapes, channels_list=channels_list_neck, num_repeats=num_repeats_neck,
        depth_mul=depth_mul, width_mul=width_mul)
    neck_out_shapes = dummy_input_run(neck, backbone_out_shapes, multi_input=True)
    neck.eval()

    head = YoloV6Head(prev_out_shape=neck_out_shapes)

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        outs = neck(outs)
        outs = head(outs)
        for i in range(len(outs)):
            print(f"Output {i}:")
            if isinstance(outs[i], list):
                for o in outs[i]:
                    print(o.shape)
            else:
                print(outs[i].shape)