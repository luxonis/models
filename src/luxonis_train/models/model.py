import torch.nn as nn
import torch
from .backbones import *
from .necks import *
from .heads import *

from luxonis_train.utils.config import Config
from luxonis_train.utils.general import dummy_input_run
from luxonis_train.utils.filesystem import LuxonisFileSystem


class Model(nn.Module):
    def __init__(self):
        """Model class for [backbone, Optional(neck), heads] architectures"""
        super(Model, self).__init__()
        self.backbone = None
        self.neck = None
        self.heads = nn.ModuleList()

    def build_model(self):
        """Builds the model from defined config"""
        cfg = Config()
        modules_cfg = cfg.get("model")
        dummy_input_shape = [
            1,
            3,
        ] + cfg.get(
            "train.preprocessing.train_image_size"
        )  # NOTE: we assume 3 dimensional input shape

        self.backbone = eval(modules_cfg["backbone"]["name"])(
            **modules_cfg["backbone"].get("params", {})
        )
        # load local backbone weights if avaliable
        if modules_cfg["backbone"]["pretrained"]:
            path = modules_cfg["backbone"]["pretrained"]
            print(f"Loading backbone weights from: {path}")
            fs = LuxonisFileSystem(path)
            checkpoint = torch.load(fs.read_to_byte_buffer())
            state_dict = checkpoint["state_dict"]
            self.backbone.load_state_dict(state_dict)

        self.backbone_out_shapes = dummy_input_run(self.backbone, dummy_input_shape)

        if "neck" in modules_cfg and modules_cfg["neck"]:
            self.neck = eval(modules_cfg["neck"]["name"])(
                prev_out_shapes=self.backbone_out_shapes,
                **modules_cfg["neck"].get("params", {}),
            )
            self.neck_out_shapes = dummy_input_run(
                self.neck, self.backbone_out_shapes, multi_input=True
            )

        for head in modules_cfg["heads"]:
            curr_head = eval(head["name"])(
                prev_out_shapes=self.neck_out_shapes
                if self.neck
                else self.backbone_out_shapes,
                original_in_shape=dummy_input_shape,
                **head["params"],
            )
            self.heads.append(curr_head)

    def forward(self, x: torch.Tensor):
        """Models forward method

        Args:
            x (torch.Tensor): Input batch

        Returns:
            outs (list): List of outputs for each models head
        """
        out = self.backbone(x)
        if self.neck != None:
            out = self.neck(out)
        outs = []
        for head in self.heads:
            curr_out = head(out)
            outs.append(curr_out)

        return outs
