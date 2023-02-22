#
# Soure:
# License:
#

import torch
import torch.nn as nn

from luxonis_train.utils.head_type import KeyPointDetection
from luxonis_train.models.heads.ikeypoint_head import IKeypoint

class YoloV7PoseHead(nn.Module):
    strides = [8., 16., 32., 64.] # possible strides. TODO: compute this in a smart way

    def __init__(self, prev_out_shape, n_classes, n_keypoints, n_layers=3, n_anchors=3, **kwargs):
        super(YoloV7PoseHead, self).__init__()

        self.n_classes = n_classes  # number of classes
        self.n_keypoints = n_keypoints
        self.type = KeyPointDetection()

        self.nl = n_layers  # number of detection layers
        self.na = n_anchors  # number of anchors

        self.head = nn.ModuleList()

        ch = 3 # number of input channels
        s = 256 # 2x min stride

        anchors = [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542]] # TODO: set or compute this
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        for i in range(self.nl):
            curr_head = IKeypoint(
                prev_out_shape=[prev_out_shape[i]],
                n_classes=self.n_classes,
                n_keypoints=self.n_keypoints,
                n_anchors=self.na
            )
            curr_head.stride = self.strides[i]
            curr_head.anchor_grid = self.anchor_grid[i]
            self.head.append(curr_head)

    def forward(self, x):
        z = []  # inference output

        for i, module in enumerate(self.head):
            out_x, out_z = module([x[i]])
            x[i] = out_x
            z.append(out_z)

        z = torch.cat(z, axis=1)
        return [x, z]

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

    head = YoloV7PoseHead(prev_out_shape=neck_out_shapes, n_classes=1, n_keypoints=17)
    head.eval()

    shapes = [224, 256, 384, 512]
    for shape in shapes:
        print("\n\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        outs = neck(outs)

        # print('NECK')
        # for i in range(len(outs)):
        #     print(f"Output {i}:")
        #     if isinstance(outs[i], list):
        #         for o in outs[i]:
        #             print(o.shape)
        #     else:
        #         print(outs[i].shape)

        outs = head(outs)

        # print('HEAD')
        for i in range(len(outs)):
            print(f"Output {i}:")
            if isinstance(outs[i], list):
                for o in outs[i]:
                    print(o.shape)
            else:
                print(outs[i].shape)
