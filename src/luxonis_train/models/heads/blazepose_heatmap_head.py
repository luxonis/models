#
# Source: https://github.com/WangChyanhassth-2say/BlazePose_torch/tree/main
# License: MIT License
#


import torch.nn as nn
from luxonis_train.models.modules import ConvModule
from luxonis_train.utils.head_type import *

class BlazePoseHeatmapHead(nn.Module):
    def __init__(self, prev_out_shape, n_classes=0, **kwargs):
        super(BlazePoseHeatmapHead, self).__init__()

        self.n_classes = n_classes
        self.type = CustomHeadType("accuracy")
        self.original_in_shape = kwargs["original_in_shape"]
        self.attach_index = kwargs.get("attach_index", -1)
        self.prev_out_shape = prev_out_shape[self.attach_index]

        self.conv = ConvModule(in_channels=self.prev_out_shape[1], out_channels=n_classes, kernel_size=3, activation=nn.Identity())

    def forward(self, x):
        # NOTE: output of this head isn't guaranteed to be same size as labels
        out = self.conv(x[self.attach_index])
        return out
    

if __name__ == "__main__":
    import torch
    from luxonis_train.models.backbones import BlazePose
    from luxonis_train.utils.general import dummy_input_run

    backbone = BlazePose()
    backbone_out_shapes = dummy_input_run(backbone, [1,3,224,224])
    backbone.eval()

    shapes = [224, 256, 384, 512]

    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = backbone(x)
        head = BlazePoseHeatmapHead(prev_out_shape=backbone_out_shapes, n_classes=17, original_in_shape=x.shape)
        head.eval()
        outs = head(outs)
        for i in range(len(outs)):
            print(f"Output {i}:")
            if isinstance(outs[i], list):
                for o in outs[i]:
                    print(len(o) if isinstance(o, list) else o.shape)
            else:
                print(outs[i].shape)