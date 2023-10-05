#
# Source: https://github.com/DingXiaoH/RepVGG
# License: https://github.com/DingXiaoH/RepVGG/blob/main/LICENSE
#


import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional, Literal

from luxonis_train.models.backbones.base_backbone import BaseBackbone
from luxonis_train.models.modules import RepVGGBlock
from luxonis_train.utils.registry import BACKBONES


@BACKBONES.register_module()
class RepVGG(BaseBackbone):
    def __init__(self, variant: Literal["A0", "A1", "A2"] = "A0", **kwargs):
        """RepVGG baackbone

        Args:
            variant (Literal["A0", "A1", "A2"], optional): Defaults to "A0".
        """
        super().__init__(**kwargs)

        if variant not in REPVGG_VARIANTS_SETTINGS.keys():
            raise ValueError(
                f"RepVGG model variant should be in {list(REPVGG_VARIANTS_SETTINGS.keys())}"
            )

        self.model = RepVGG_(**REPVGG_VARIANTS_SETTINGS[variant])

    def forward(self, x):
        features = self.model(x)
        return features


class RepVGG_(nn.Module):
    def __init__(
        self,
        num_blocks: list,
        num_classes: int = 1000,
        width_multiplier: Optional[list] = None,
        override_groups_map: Optional[list] = None,
        deploy: bool = False,
        use_se: bool = False,
        use_checkpoint: bool = False,
    ):
        super(RepVGG_, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_se=self.use_se,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                    use_se=self.use_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        outputs = []
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
            outputs.append(out)
        return outputs


REPVGG_VARIANTS_SETTINGS = {
    "A0": {
        "num_blocks": [2, 4, 14, 1],
        "num_classes": 1000,
        "width_multiplier": [0.75, 0.75, 0.75, 2.5],
        "override_groups_map": None,
    },
    "A1": {
        "num_blocks": [2, 4, 14, 1],
        "num_classes": 1000,
        "width_multiplier": [1, 1, 1, 2.5],
        "override_groups_map": None,
    },
    "A2": {
        "num_blocks": [2, 4, 14, 1],
        "num_classes": 1000,
        "width_multiplier": [1.5, 1.5, 1.5, 2.75],
        "override_groups_map": None,
    },
}
