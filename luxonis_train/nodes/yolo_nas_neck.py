"""Yolo NAS neck.

YoloNASPANNeckWithC2
Source: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/yolo_nas/panneck.py
License: Apache-2.0 license https://github.com/Deci-AI/super-gradients?tab=Apache-2.0-1-ov-file#readme
Note: Only the super-gradients source code is Apache-2.0, all Yolo NAS & Yolo NAS Pose weights are under a restrictive license
Weights license: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md
"""


import torchvision
from torch import Tensor
import torch.nn as nn

from .base_node import BaseNode
from luxonis_train.nodes.blocks.yolo_nas_blocks import (
    YoloNASUpStage,
    YoloNASDownStage
)

from typing import List


class YoloNASNeck(BaseNode[list[Tensor], list[Tensor]]):
    attach_index: int = -1

    VARIANTS_CONFIGS: dict[str, dict] = {
        "n": None,
        "s": {
            "modules": {
                "neck_0": {
                    "out_channels": 192,
                    "num_blocks": 2,
                    "hidden_channels": 64,
                    "width_mult": 1,
                    "depth_mult": 1,
                    "activation_type": nn.ReLU,
                    "reduce_channels": True
                },
                "neck_1": {
                    "out_channels": 96,
                    "num_blocks": 2,
                    "hidden_channels": 48,
                    "width_mult": 1,
                    "depth_mult": 1,
                    "activation_type": nn.ReLU,
                    "reduce_channels": True
                },
                "neck_2": {
                    "out_channels": 192,
                    "num_blocks": 2,
                    "hidden_channels": 64,
                    "activation_type": nn.ReLU,
                    "width_mult": 1,
                    "depth_mult": 1
                },
                "neck_3": {
                    "out_channels": 384,
                    "num_blocks": 2,
                    "hidden_channels": 64,
                    "activation_type": nn.ReLU,
                    "width_mult": 1,
                    "depth_mult": 1
                }
            },
            "in_channels": [96, 192, 384, 768]
        },
        "m": None,
        "l": None,
    }

    def __init__(
        self,
        variant: str,
        **kwargs,
    ):
        """Simple wrapper for YoloNASPANNeckWithC2 neck, source above ^^.

        Args:
            variant (str): Yolo NAS variant ["n", "s", "m", "l"].
        """
        super().__init__(**kwargs)

        if not variant in YoloNASNeck.VARIANTS_CONFIGS:
            raise ValueError(
                f"YoloNASNeck variant should be in {YoloNASNeck.VARIANTS_CONFIGS.keys()}"
            )
        
        self.variant_config = YoloNASNeck.VARIANTS_CONFIGS[variant]
        c2_out_channels, c3_out_channels, c4_out_channels, c5_out_channels = self.variant_config["in_channels"]

        self.neck_0 = YoloNASUpStage(
            in_channels=[c5_out_channels, c4_out_channels, c3_out_channels], 
            **self.variant_config["modules"]["neck_0"]     
        )
        self.neck_1 = YoloNASUpStage(
            in_channels=[self.neck_0.out_channels[1], c3_out_channels, c2_out_channels], 
            **self.variant_config["modules"]["neck_1"]     
        )
        self.neck_2 = YoloNASDownStage(
            in_channels=[self.neck_1.out_channels[1], self.neck_1.out_channels[0]], 
            **self.variant_config["modules"]["neck_2"]     
        )
        self.neck_3 = YoloNASDownStage(
            in_channels=[self.neck_2.out_channels, self.neck_0.out_channels[0]], 
            **self.variant_config["modules"]["neck_3"]     
        )

        self._out_channels = [
            self.neck_1.out_channels[1],
            self.neck_2.out_channels,
            self.neck_3.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs

        x_n1_inter, x = self.neck_0([c5, c4, c3])
        x_n2_inter, p3 = self.neck_1([x, c3, c2])
        p4 = self.neck_2([p3, x_n2_inter])
        p5 = self.neck_3([p4, x_n1_inter])

        return p3, p4, p5
