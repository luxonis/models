import torch.nn as nn
import torch
from .backbones import *
from .necks import *
from .heads import *
from luxonis_train.utils.general import dummy_input_run

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = None
        self.neck = None
        self.heads = nn.ModuleList()

    def build_model(self, modules, input_shape):
        self.modules_cfg = modules
        self.input_shape = input_shape

        self.backbone = eval(self.modules_cfg["backbone"]["name"]) \
            (**self.modules_cfg["backbone"]["params"] if self.modules_cfg["backbone"]["params"] else {})
        # load local backbone weights if avaliable
        if self.modules_cfg["backbone"]["pretrained"]:
            self.backbone.load_state_dict(torch.load(self.modules_cfg["backbone"]["pretrained"])["state_dict"])

        self.backbone_out_shapes = dummy_input_run(self.backbone, [1,3,]+self.input_shape) # TODO: we assume 3 dimensional input shape
        
        if self.modules_cfg["neck"]:
            self.neck = eval(self.modules_cfg["neck"]["name"])(
                    prev_out_shape = self.backbone_out_shapes,
                    **self.modules_cfg["neck"]["params"] if self.modules_cfg["neck"]["params"] else {}
                )
            self.neck_out_shapes = dummy_input_run(self.neck, self.backbone_out_shapes, multi_input=True)

        for head in self.modules_cfg["heads"]:
            curr_head = eval(head["name"])(
                    prev_out_shape = self.neck_out_shapes if self.neck else self.backbone_out_shapes,
                    original_in_shape = [1,3]+self.input_shape,
                    **head["params"] if head["params"] else {},
                )
            self.heads.append(curr_head)

    def forward(self, x):
        out = self.backbone(x)
        if self.neck != None:
            out = self.neck(out)
        outs = []
        for head in self.heads:
            curr_out = head(out)
            outs.append(curr_out)

        return outs