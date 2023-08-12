#
# Soure: https://github.com/clovaai/rexnet
# License: https://github.com/clovaai/rexnet/blob/master/LICENSE
#


import torch
import torch.nn as nn

from luxonis_train.utils.general import make_divisible
from luxonis_train.models.modules import ConvModule


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, kernel_size=3, stride=1, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.conv_shortcut = None
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        out = []
        if t != 1:
            dw_channels = in_channels * t
            out.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=dw_channels,
                    kernel_size=1,
                    activation=nn.ReLU6(inplace=True),
                )
            )
        else:
            dw_channels = in_channels
        out.append(
            ConvModule(
                in_channels=dw_channels,
                out_channels=dw_channels * 1,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size // 2),
                groups=dw_channels,
                activation=nn.ReLU6(inplace=True),
            )
        )
        out.append(
            ConvModule(
                in_channels=dw_channels,
                out_channels=channels,
                kernel_size=1,
                activation=nn.Identity(),
            )
        )

        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)

        if self.use_shortcut:
            # this results in a ScatterND node which isn't supported yet in myriad
            # out[:, 0:self.in_channels] += x

            a = out[:, : self.in_channels]
            b = x
            a = a + b
            c = out[:, self.in_channels :]
            d = torch.concat([a, c], dim=1)
            return d
        return out


class ReXNetV1_lite(nn.Module):
    def __init__(
        self,
        fix_head_stem: bool = False,
        divisible_value: int = 8,
        input_ch: int = 16,
        final_ch: int = 164,
        multiplier: float = 1.0,
        kernel_conf: str = "333333",
    ):
        """ReXNetV1_lite backbone

        Args:
            fix_head_stem (bool, optional): Weather to multiply head stem. Defaults to False.
            divisible_value (int, optional): Divisor used. Defaults to 8.
            input_ch (int, optional): Starting channel dimension. Defaults to 16.
            final_ch (int, optional): Final channel dimension. Defaults to 164.
            multiplier (float, optional): Channel dimension multiplier. Defaults to 1.0.
            kernel_conf (str, optional): Kernel sizes encoded as string. Defaults to '333333'.
        """
        super().__init__()

        self.out_indices = [1, 4, 10, 16]
        self.channels = [16, 48, 112, 184]
        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        kernel_sizes = [int(element) for element in kernel_conf]

        strides = sum(
            [
                [element] + [1] * (layers[idx] - 1)
                for idx, element in enumerate(strides)
            ],
            [],
        )
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        kernel_sizes = sum(
            [[element] * layers[idx] for idx, element in enumerate(kernel_sizes)], []
        )
        self.num_convblocks = sum(layers[:])

        features = []
        inplanes = input_ch / multiplier if multiplier < 1.0 else input_ch
        first_channel = 32 / multiplier if multiplier < 1.0 or fix_head_stem else 32
        first_channel = make_divisible(
            int(round(first_channel * multiplier)), divisible_value
        )

        in_channels_group = []
        channels_group = []

        features.append(
            ConvModule(
                3,
                first_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.ReLU6(inplace=True),
            )
        )

        for i in range(self.num_convblocks):
            inplanes_divisible = make_divisible(
                int(round(inplanes * multiplier)), divisible_value
            )
            if i == 0:
                in_channels_group.append(first_channel)
                channels_group.append(inplanes_divisible)
            else:
                in_channels_group.append(inplanes_divisible)
                inplanes += final_ch / (self.num_convblocks - 1 * 1.0)
                inplanes_divisible = make_divisible(
                    int(round(inplanes * multiplier)), divisible_value
                )
                channels_group.append(inplanes_divisible)

        for block_idx, (in_c, c, t, k, s) in enumerate(
            zip(in_channels_group, channels_group, ts, kernel_sizes, strides)
        ):
            features.append(
                LinearBottleneck(
                    in_channels=in_c, channels=c, t=t, kernel_size=k, stride=s
                )
            )

        pen_channels = (
            int(1280 * multiplier) if multiplier > 1 and not fix_head_stem else 1280
        )
        features.append(
            ConvModule(
                in_channels=c,
                out_channels=pen_channels,
                kernel_size=1,
                activation=nn.ReLU6(inplace=True),
            )
        )
        self.features = nn.Sequential(*features)

    def forward(self, x):
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if __name__ == "__main__":
    model = ReXNetV1_lite(multiplier=1.0)
    model.eval()

    shapes = [224, 256, 384, 512]

    for shape in shapes:
        print("\nShape", shape)
        x = torch.zeros(1, 3, shape, shape)
        outs = model(x)
        if isinstance(outs, list):
            for out in outs:
                print(out.shape)
        else:
            print(outs.shape)
