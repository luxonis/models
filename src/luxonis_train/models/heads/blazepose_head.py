#
# Source: https://github.com/WangChyanhassth-2say/BlazePose_torch/tree/main
# License: MIT License
#


import torch.nn as nn
from luxonis_train.utils.head_type import *
from luxonis_train.models.modules import MobileBottleneck
from luxonis_train.models.backbones.blazepose import BlazeBlock

class BlazePoseHead(nn.Module):
    def __init__(self, prev_out_shape, n_classes=0, **kwargs):
        super(BlazePoseHead, self).__init__()

        self.n_classes = n_classes
        self.type = KeyPointDetection()
        self.original_in_shape = kwargs["original_in_shape"]
        self.attach_index = kwargs.get("attach_index", -1)
        self.prev_out_shape = prev_out_shape[self.attach_index]

        self.conv12_a1 = MobileBottleneck(32, 32, 5, 1, 120, True, nn.ReLU6())
        self.conv13_a1 = MobileBottleneck(64, 64, 3, 1, 200, False, nn.SiLU())
        self.conv13_a2 = MobileBottleneck(64, 64, 3, 1, 184, False, nn.SiLU())
        self.conv14_a1 = MobileBottleneck(128, 128, 3, 1, 480, True, nn.SiLU())
        self.conv14_a2 = MobileBottleneck(128, 128, 3, 1, 672, True, nn.SiLU())
        self.conv15_a1 = MobileBottleneck(192, 192, 5, 1, 960, True, nn.SiLU())

        self.conv12 = BlazeBlock(32, 64, stride=2)
        self.conv13 = BlazeBlock(64, 128, stride=2)
        self.conv14 = BlazeBlock(128, 192, stride=2)

        self.conv15 = nn.Sequential(
            nn.Conv2d(192, self.n_classes, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.mlp_head_x = nn.Linear(64, int(256 * 2))
        self.mlp_head_y = nn.Linear(64, int(256 * 2))

    def forward(self, x):
        y0, y1, y2, y3, x0 = x
        out = x0 + y0
        out = self.conv12_a1(out)
        out = self.conv12(out) + y1
        out = self.conv13_a1(out)
        out = self.conv13_a2(out)
        out = self.conv13(out) + y2
        out = self.conv14_a1(out)
        out = self.conv14_a2(out)
        out = self.conv14(out) + y3
        out = self.conv15_a1(out)
        out = self.conv15(out)
        # x = rearrange(x, 'b c h w -> b c (h w)')
        # pred_x = self.mlp_head_x(x)
        # pred_y = self.mlp_head_y(x)
        # return pred_x, pred_y
        return out

if __name__ == "__main__":
    import torch
    from luxonis_train.models.backbones import BlazePose
    from luxonis_train.utils.general import dummy_input_run

    backbone = BlazePose()
    backbone_out_shapes = dummy_input_run(backbone, [1,3,224,224])
    backbone.eval()

    shapes = [224]
    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        head = BlazePoseHead(prev_out_shape=backbone_out_shapes, n_classes=17, original_in_shape=x.shape)
        head.eval()
        print(head)
        break
        outs = head(outs)
        for i in range(len(outs)):
            print(f"Output {i}:")
            if isinstance(outs[i], list):
                for o in outs[i]:
                    print(len(o) if isinstance(o, list) else o.shape)
            else:
                print(outs[i].shape)