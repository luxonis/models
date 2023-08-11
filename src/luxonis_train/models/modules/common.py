import torch
import torch.nn as nn
from typing import Optional, Union
import numpy as np

from .activations import *


class ConvModule(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        activation: Optional[object] = nn.ReLU(),
    ):
        """Conv2d + BN + Activation

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (Optional[int], optional): Defaults to 1.
            padding (Optional[int], optional): Defaults to 0.
            dilation (Optional[int], optional): Defaults to 1.
            groups (Optional[int], optional): Defaults to 1.
            bias (Optional[bool], optional): Defaults to False.
            activation (Optional[object], optional): Defaults to nn.ReLU().
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
        kernel_size: Optional[int] = 2,
        stride: Optional[int] = 2,
    ):
        """Upsampling with ConvTranspose2D (similar to U-Net Up block)

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (Optional[int], optional): Defaults to 2.
            stride (Optional[int], optional): Defaults to 2.
        """
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1),
        )


class SEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        approx_sigmoid: Optional[bool] = False,
        activation: Optional[object] = nn.ReLU(),
    ):
        """Squeeze and Excite block. Adapted from: https://github.com/apple/ml-mobileone/blob/main/mobileone.py

        Args:
            in_channels (int): Number of input channels
            intermediate_channels (int): Number of intermediate channels
            approx_sigmoid (Optional[bool], optional): Whether to use approximated sigmoid function. Defaults to False.
            activation (Optional[object], optional): Defaults to nn.ReLU().
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
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 1,
        padding: Optional[int] = 1,
        dilation: Optional[int] = 1,
        groups: Optional[int] = 1,
        padding_mode: Optional[str] = "zeros",
        deploy: Optional[bool] = False,
        use_se: Optional[bool] = False,
    ):
        """RepVGGBlock is a basic rep-style block, including training and deploy status
        This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (Optional[int], optional): Defaults to 3.
            stride (Optional[int], optional): Defaults to 1.
            padding (Optional[int], optional): Defaults to 1.
            dilation (Optional[int], optional): Defaults to 1.
            groups (Optional[int], optional): Defaults to 1.
            padding_mode (Optional[str], optional): Defaults to "zeros".
            deploy (Optional[bool], optional): Defaults to False.
            use_se (Optional[bool], optional): Weather to use SEBlock. Defaults to False.
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
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock(
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
    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: Optional[int] = 1
    ):
        """Module which consists of multiple RepVGGBlocks

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (Optional[int], optional): Number of RepVGG blocks. Defaults to 1.
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


class SimplifiedSPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SimplifiedSPPF, self).__init__()
        """ Simplified Spatial Pyramid Pooling with ReLU activation 
            Adapted from: https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/layers/common.py
        """
        c_ = in_channels // 2  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, 1)
        self.cv2 = ConvModule(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        # Pass the input feature map through the first convolutional layer
        x = self.cv1(x)

        # apply max-pooling at three different scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # Concatenate the original feature map and the three max-pooled versions
        # along the channel dimension and pass through the second convolutional layer
        out = self.cv2(torch.cat([x, y1, y2, y3], dim=1))
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
