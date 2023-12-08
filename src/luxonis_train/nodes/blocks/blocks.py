# TODO:  cleanup, document
# Check if some blocks could be merged togetner.

import math
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor, nn

from luxonis_train.nodes.activations import HSigmoid


class EfficientDecoupledBlock(nn.Module):
    def __init__(self, n_classes: int, in_channels: int):
        """Efficient Decoupled block used for class and regression predictions.

        Args:
            n_classes (int): Number of classes
            in_channels (int): Number of input channels
        """
        super().__init__()

        self.decoder = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            activation=nn.SiLU(),
        )

        self.class_branch = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(in_channels=in_channels, out_channels=n_classes, kernel_size=1),
        )
        self.regression_branch = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1),
        )

        prior_prob = 1e-2
        self._initialize_weights_and_biases(prior_prob)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out_feature = self.decoder(x)

        out_cls = self.class_branch(out_feature)
        out_reg = self.regression_branch(out_feature)

        return out_feature, out_cls, out_reg

    def _initialize_weights_and_biases(self, prior_prob: float):
        data = [
            (self.class_branch[-1], -math.log((1 - prior_prob) / prior_prob)),
            (self.regression_branch[-1], 1.0),
        ]
        for module, fill_value in data:
            assert module.bias is not None
            b = module.bias.view(-1)
            b.data.fill_(fill_value)
            module.bias = nn.Parameter(b.view(-1), requires_grad=True)

            w = module.weight
            w.data.fill_(0.0)
            module.weight = nn.Parameter(w, requires_grad=True)


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
        activation: nn.Module | None = None,
    ):
        """Conv2d + BN + Activation.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 0.
            dilation (int, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            bias (bool, optional): Defaults to False.
            activation (nn.Module, optional): Defaults to nn.ReLU().
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
            activation or nn.ReLU(),
        )


class UpBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        """Upsampling with ConvTranspose2D (similar to U-Net Up block).

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
        activation: nn.Module | None = None,
    ):
        """Squeeze and Excite block from `Squeeze-and-Excitation Networks`,
            https://arxiv.org/pdf/1709.01507.pdf. Adapted from: https://github.com/apple/ml-mobileone/blob/main/mobileone.py.

        Args:
            in_channels (int): Number of input channels
            intermediate_channels (int): Number of intermediate channels
            approx_sigmoid (bool, optional): Whether to use approximated sigmoid function. Defaults to False.
            activation (nn.Module, optional): Defaults to nn.ReLU().
        """
        super().__init__()

        activation = activation or nn.ReLU()
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

    def forward(self, x: Tensor) -> Tensor:
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
        This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

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
            use_se (bool, optional): Whether to use SqueezeExciteBlock. Defaults to False.
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
            self.se = nn.Identity()  # type: ignore

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

    def forward(self, x: Tensor):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + id_out))

    def reparametrize(self):
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
        self.rbr_reparam.weight.data = kernel  # type: ignore
        self.rbr_reparam.bias.data = bias  # type: ignore
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

    def _get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a DIFFERENTIABLE way."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3
            + self._pad_1x1_to_3x3_tensor(kernel1x1)
            + kernelid.to(kernel3x3.device),
            bias3x3 + bias1x1 + biasid.to(bias3x3.device),
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: Tensor | None) -> Tensor:
        if kernel1x1 is None:
            return torch.tensor(0)
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module | None) -> tuple[Tensor, Tensor]:
        if branch is None:
            return torch.tensor(0), torch.tensor(0)
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
        assert running_var is not None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1).to(kernel.device)
        return kernel * t, beta - running_mean * gamma / std


class BlockRepeater(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
    ):
        """Module which repeats the block n times. First block accepts in_channels and
        outputs out_channels while subsequent blocks accept out_channels and output
        out_channels.

        Args:
            block (nn.Module): Block to repeat
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int] optional): Number of RepVGG blocks. Defaults to 1.
        """
        super().__init__()

        in_channels = in_channels
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                block(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SpatialPyramidPoolingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        """Spatial Pyramid Pooling block with ReLU activation on three different scales.

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


class LearnableAdd(nn.Module):
    def __init__(self, channel: int):
        """Implicit add block."""
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self, x: Tensor):
        return self.implicit.expand_as(x) + x


class LearnableMultiply(nn.Module):
    def __init__(self, channel: int):
        """Implicit multiply block."""
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x: Tensor):
        return self.implicit.expand_as(x) * x


class LearnableMulAddConv(nn.Module):
    def __init__(
        self,
        add_channel: int,
        mul_channel: int,
        conv_in_channel: int,
        conv_out_channel: int,
    ):
        super().__init__()
        self.add = LearnableAdd(add_channel)
        self.mul = LearnableMultiply(mul_channel)
        self.conv = nn.Conv2d(conv_in_channel, conv_out_channel, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.mul(self.conv(self.add(x)))


class KeypointBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Keypoint head block for keypoint predictions."""
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(6):
            depth_wise_conv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=autopad(3),
                groups=math.gcd(in_channels, in_channels),
                activation=nn.SiLU(),
            )
            conv = (
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    padding=autopad(1),
                    activation=nn.SiLU(),
                )
                if i < 5
                else nn.Conv2d(in_channels, out_channels, 1)
            )

            layers.append(depth_wise_conv)
            layers.append(conv)

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        out = self.block(x)
        return out


class RepUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_channels_next: int,
        out_channels: int,
        num_repeats: int,
    ):
        """UpBlock used in RepPAN neck.

        Args:
            in_channels (int): Number of input channels
            in_channels_next (int): Number of input channels of next input
            which is used in concat
            out_channels (int): Number of output channels
            num_repeats (int): Number of RepVGGBlock repeats
        """
        super().__init__()

        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.rep_block = BlockRepeater(
            block=RepVGGBlock,
            in_channels=in_channels_next + out_channels,
            out_channels=out_channels,
            num_blocks=num_repeats,
        )

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        conv_out = self.conv(x0)
        upsample_out = self.upsample(conv_out)
        concat_out = torch.cat([upsample_out, x1], dim=1)
        out = self.rep_block(concat_out)
        return conv_out, out


class RepDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        in_channels_next: int,
        out_channels: int,
        num_repeats: int,
    ):
        """DownBlock used in RepPAN neck.

        Args:
            in_channels (int): Number of input channels
            downsample_out_channels (int): Number of output channels after downsample
            in_channels_next (int): Number of input channels of next input which
            is used in concat
            out_channels (int): Number of output channels
            num_repeats (int): Number of RepVGGBlock repeats
        """
        super().__init__()

        self.downsample = ConvModule(
            in_channels=in_channels,
            out_channels=downsample_out_channels,
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )
        self.rep_block = BlockRepeater(
            block=RepVGGBlock,
            in_channels=downsample_out_channels + in_channels_next,
            out_channels=out_channels,
            num_blocks=num_repeats,
        )

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        x = self.downsample(x0)
        x = torch.cat([x, x1], dim=1)
        x = self.rep_block(x)
        return x


T = TypeVar("T", int, tuple[int, ...])


def autopad(kernel_size: T, padding: T | None = None) -> T:
    """Compute padding based on kernel size.

    Args:
        kernel_size (int | tuple[int, ...]): The kernel size
        padding (int | tuple[int, ...], optional): The defalt padding value
          or a tuple of padding values. Defaults to None.
          Will be directly returned if specified.

    Returns:
        int | tuple: The computed padding value(s). The output type is the same
          as the type of the `kernel_size`.
    """
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(x // 2 for x in kernel_size)