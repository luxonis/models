import torch
import torch.nn as nn
from typing import Union
import numpy as np

from .activations import *


class ConvModule(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: object = nn.ReLU(),
    ):
        """Conv2d + BN + Activation

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 0.
            dilation (int, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            bias (bool, optional): Defaults to False.
            activation (object, optional): Defaults to nn.ReLU().
        """
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )


class UpBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        """Upsampling with ConvTranspose2D (similar to U-Net Up block)

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int] optional): Defaults to 2.
            stride (int] optional): Defaults to 2.
        """
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1),
        )


class SqueezeExciteBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        approx_sigmoid: bool = False,
        activation: object = nn.ReLU(),
    ):
        """Squeeze and Excite block from `Squeeze-and-Excitation Networks`,
            https://arxiv.org/pdf/1709.01507.pdf. Adapted from: https://github.com/apple/ml-mobileone/blob/main/mobileone.py

        Args:
            in_channels (int): Number of input channels
            intermediate_channels (int): Number of intermediate channels
            approx_sigmoid (bool, optional): Whether to use approximated sigmoid function. Defaults to False.
            activation (object, optional): Defaults to nn.ReLU().
        """
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            bias=True,
        )
        self.activation = activation
        self.conv_up = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True,
        )
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        weights = self.pool(x)
        weights = self.conv_down(weights)
        weights = self.activation(weights)
        weights = self.conv_up(weights)
        weights = self.sigmoid(weights)
        x = x * weights
        return x


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
        use_se: bool = False,
    ):
        """RepVGGBlock is a basic rep-style block, including training and deploy status
        This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Defaults to 3.
            stride (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
            dilation (int, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            padding_mode (str, optional): Defaults to "zeros".
            deploy (bool, optional): Defaults to False.
            use_se (bool, optional): Weather to use SqueezeExciteBlock. Defaults to False.
        """
        super().__init__()

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SqueezeExciteBlock after nonlinearity.
            self.se = SqueezeExciteBlock(
                out_channels, intermediate_channels=int(out_channels // 16)
            )
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                activation=nn.Identity(),
            )
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                activation=nn.Identity(),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        )

    def to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            dilation=self.rbr_dense[0].dilation,
            groups=self.rbr_dense[0].groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

    def _get_equivalent_kernel_bias(self):
        """derives the equivalent kernel and bias in a DIFFERENTIABLE way"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
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
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepVGGBlockN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        """Module which consists of multiple RepVGGBlocks

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int] optional): Number of RepVGG blocks. Defaults to 1.
        """
        super().__init__()

        in_channels = in_channels
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                RepVGGBlock(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SpatialPyramidPoolingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        """Spatial Pyramid Pooling block with ReLU activation
        Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/layers/common.py

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int] optional): Defaults to 5.
        """
        super().__init__()

        intermediate_channels = in_channels // 2  # hidden channels
        self.conv1 = ConvModule(in_channels, intermediate_channels, 1, 1)
        self.conv2 = ConvModule(intermediate_channels * 4, out_channels, 1, 1)
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.conv1(x)
        # apply max-pooling at three different scales
        y1 = self.max_pool(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)

        x = torch.cat([x, y1, y2, y3], dim=1)
        x = self.conv2(x)
        return x


class AttentionRefinmentBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        """Attention Refinment block adapted from: https://github.com/taveraantonio/BiseNetv1 """

        self.conv_3x3 = ConvModule(in_channels, out_channels, 3, 1, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation=nn.Identity(),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_3x3(x)
        attention = self.attention(x)
        out = x * attention
        return out


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1):
        super().__init__()
        """Feature Fusion block adapted from: https://github.com/taveraantonio/BiseNetv1 """

        self.conv_1x1 = ConvModule(in_channels, out_channels, 1, 1, 0)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels // reduction,
                kernel_size=1,
            ),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels // reduction,
                kernel_size=1,
                activation=nn.Identity(),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        x = self.conv_1x1(fusion)
        attention = self.attention(x)
        out = x + x * attention
        return out


def autopad(k: Union[int, tuple], p: Union[int, tuple] = None):
    """Compute padding based on kernel size.

    Args:
        k (Union[int, tuple]): The kernel size
        p (Union[int, tuple], optional): The padding value or tuple of padding values. Defaults to None.

    Returns:
        Union[int, tuple]: The computed padding value(s).
    """
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = tuple(x // 2 for x in k)  # auto-pad for each dimension
    return p
