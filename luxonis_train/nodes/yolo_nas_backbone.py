"""Yolo NAS backbone.

NStageBackbone
Source: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/modules/detection_modules.py 
License: Apache-2.0 license https://github.com/Deci-AI/super-gradients?tab=Apache-2.0-1-ov-file#readme
Note: Only the super-gradients source code is Apache-2.0, all Yolo NAS & Yolo NAS Pose weights are under a restrictive license
Weights license: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md
"""


import torchvision
from torch import Tensor
import torch.nn as nn

from .base_node import BaseNode
from luxonis_train.nodes.blocks.yolo_nas_blocks import (
    QARepVGGBlock,
    YoloNASStage,
    SPP
)

from typing import OrderedDict


class YoloNASBackbone(BaseNode[Tensor, list[Tensor]]):
    attach_index: int = -1

    VARIANTS_CONFIGS: dict[str, dict] = {
        "n": None,
        "s": {
            "modules": {
                "stem": {
                    "out_channels": 48, 
                    "stride": 2, 
                    "is_residual": False
                },
                "stage_0": {
                    "in_channels": 48,
                    "out_channels": 96,
                    "num_blocks": 2,
                    "activation_type": nn.ReLU,
                    "hidden_channels": 32,
                    "concat_intermediates": False
                },
                "stage_1": {
                    "in_channels": 96,
                    "out_channels": 192,
                    "num_blocks": 3,
                    "activation_type": nn.ReLU,
                    "hidden_channels": 64,
                    "concat_intermediates": False
                },
                "stage_2": {
                    "in_channels": 192,
                    "out_channels": 384,
                    "num_blocks": 5,
                    "activation_type": nn.ReLU,
                    "hidden_channels": 96,
                    "concat_intermediates": False
                },
                "stage_3": {
                    "in_channels": 384,
                    "out_channels": 768,
                    "num_blocks": 2,
                    "activation_type": nn.ReLU,
                    "hidden_channels": 192,
                    "concat_intermediates": False
                },
                "context": {
                    "in_channels": 768,
                    "out_channels": 768,
                    "k": [5,9,13]
                }
            },
            "out_layers": ["stage_0", "stage_1", "stage_2", "context"]
        },
        "m": None,
        "l": None,
    }

    def __init__(
        self,
        in_channels: int,
        variant: str,
        **kwargs,
    ):
        """Simple wrapper for NStageBackbone backbone, source above ^^.

        Args:
            in_channels (int): number of input channels, 3 for RGB/BGR & 1 for MONO.
            variant (str): Yolo NAS variant ["n", "s", "m", "l"].
        """
        super().__init__(**kwargs)

        if not variant in YoloNASBackbone.VARIANTS_CONFIGS:
            raise ValueError(
                f"YoloNASBackbone variant should be in {YoloNASBackbone.VARIANTS_CONFIGS.keys()}"
            )
        
        self.variant_config = YoloNASBackbone.VARIANTS_CONFIGS[variant]
        self.backbone = nn.Sequential(
            OrderedDict([
                ("stem", QARepVGGBlock(
                    in_channels=in_channels, 
                    **self.variant_config["modules"]["stem"]
                )),
                ("stage_0", YoloNASStage(
                    **self.variant_config["modules"]["stage_0"]
                )),
                ("stage_1", YoloNASStage(
                    **self.variant_config["modules"]["stage_1"]
                )),
                ("stage_2", YoloNASStage(
                    **self.variant_config["modules"]["stage_2"]
                )),
                ("stage_3", YoloNASStage(
                    **self.variant_config["modules"]["stage_3"]
                )),
                ("context", SPP(
                    in_channels=768,
                    out_channels=768,
                    k=[5,9,13]
                )),
                
            ])
        )
        self.out_layers = self.variant_config["out_layers"]

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []

        for name, module in self.backbone.named_children():
            x = module(x)
            if name in self.out_layers:
                outs.append(x)

        return outs
