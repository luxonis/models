#
# Soure: https://github.com/apple/ml-mobileone
# License: https://github.com/apple/ml-mobileone/blob/main/LICENSE
#


import torch
import torch.nn as nn
from typing import List, Literal
import torch
import torch.nn as nn

from luxonis_train.models.backbones.base_backbone import BaseBackbone
from luxonis_train.models.modules import SqueezeExciteBlock, ConvModule
from luxonis_train.utils.registry import BACKBONES


@BACKBONES.register_module()
class MobileOne(BaseBackbone):
    def __init__(self, variant: Literal["s0", "s1", "s2", "s3", "s4"] = "s0", **kwargs):
        """MobileOne backbone

        Args:
            variant (Literal["s0", "s1", "s2", "s3", "s4"], optional): Defaults to "s0".
        """
        super().__init__(**kwargs)

        if variant not in MOBILEONE_VARIANTS_SETTINGS.keys():
            raise ValueError(
                f"MobileOne model variant should be in {list(MOBILEONE_VARIANTS_SETTINGS.keys())}"
            )

        variant_params = MOBILEONE_VARIANTS_SETTINGS[variant]
        self.backbone = MobileOne_(**variant_params)

    def forward(self, x):
        outs = []
        x = self.backbone.stage0(x)
        outs.append(x)
        x = self.backbone.stage1(x)
        outs.append(x)
        x = self.backbone.stage2(x)
        outs.append(x)
        x = self.backbone.stage3(x)
        outs.append(x)

        return outs

    def to_deploy(self):
        """Switch modules of the model to deploy"""
        for module in self.backbone.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()


class MobileOne_(nn.Module):
    """MobileOne Model
    Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        num_blocks_per_stage: List[int] = [2, 8, 10, 1],
        width_multipliers: List[float] = None,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ):
        """Construct MobileOne model.

        Args:
            num_blocks_per_stage (List[int], optional): List of number of blocks per stage. Defaults to [2, 8, 10, 1].
            width_multipliers (List[float], optional): List of width multiplier for blocks in a stage.. Defaults to None.
            use_se (bool, optional): Whether to use SE-ReLU activations.. Defaults to False.
            num_conv_branches (int, optional): Number of linear conv branches.. Defaults to 1.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
        )
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
        )
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
        )

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int):
        """Build a stage of MobileOne model.

        Args:
            planes (int): Number of output channels.
            num_blocks (int): Number of blocks in this stage.
            num_se_blocks (int): Number of SE blocks in this stage.

        Returns:
            nn.Sequential: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    "Number of SE blocks cannot " "exceed number of layers."
                )
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            # Pointwise conv
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        """Apply forward pass."""
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class MobileOneBlock(nn.Module):
    """MobileOne building block.
    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_se: bool = False,
        num_conv_branches: int = 1,
    ):
        """Construct a MobileOneBlock module.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the block.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride size. Defaults to 1.
            padding (int, optional): Zero-padding size. Defaults to 0.
            dilation (int, optional): Kernel dilation factor. Defaults to 1.
            groups (int, optional): Group number. Defaults to 1.
            use_se (bool, optional): Whether to use SE-ReLU activations. Defaults to False.
            num_conv_branches (int, optional): Number of linear conv branches. Defaults to 1.
        """

        super(MobileOneBlock, self).__init__()

        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.inference_mode = False

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SqueezeExciteBlock(
                in_channels=out_channels, internal_channels=int(out_channels * 0.0625)
            )
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        # Re-parameterizable skip connection
        self.rbr_skip = (
            nn.BatchNorm2d(num_features=in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )

        # Re-parameterizable conv branches
        rbr_conv = list()
        for _ in range(self.num_conv_branches):
            rbr_conv.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=padding,
                    groups=self.groups,
                    activation=nn.Identity(),
                )
            )
        self.rbr_conv = nn.ModuleList(rbr_conv)

        # Re-parameterizable scale branch
        self.rbr_scale = None
        if kernel_size > 1:
            self.rbr_scale = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                groups=self.groups,
                activation=nn.Identity(),
            )

    def forward(self, x: torch.Tensor):
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        print("TEST")
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0][0].in_channels,
            out_channels=self.rbr_conv[0][0].out_channels,
            kernel_size=self.rbr_conv[0][0].kernel_size,
            stride=self.rbr_conv[0][0].stride,
            padding=self.rbr_conv[0][0].padding,
            dilation=self.rbr_conv[0][0].dilation,
            groups=self.rbr_conv[0][0].groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self):
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


MOBILEONE_VARIANTS_SETTINGS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
}
