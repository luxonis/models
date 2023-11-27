"""BiSeNet segmentation head.

Adapted from `https://github.com/taveraantonio/BiseNetv1`.
License: NOT SPECIFIED.
"""

from torch import Tensor, nn
from typeguard import check_type

from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.utils.general import infer_upscale_factor
from luxonis_train.utils.types import Packet

from .luxonis_node import LuxonisNode


class BiSeNetHead(LuxonisNode[Tensor, Tensor]):
    """Implementation of the BiSeNet segmentation head.

    TODO: Add more documentation.
    """

    def __init__(
        self,
        intermediate_channels: int = 64,
        **kwargs,
    ):
        """Constructor for the BiSeNet segmentation head.

        Args:
            n_classes (int): Number of classes
            intermediate_channels (int, optional): How many intermediate channels to
              use. Defaults to 64.
        """
        super().__init__(attach_index=-1, **kwargs)

        original_height = self.original_in_shape[2]
        upscale_factor = 2 ** infer_upscale_factor(
            check_type(self.in_height, int), original_height
        )
        out_channels = self.dataset_metadata.n_classes * upscale_factor * upscale_factor

        self.conv_3x3 = ConvModule(
            check_type(self.in_channels, int),
            intermediate_channels,
            3,
            1,
            1,
        )
        self.conv_1x1 = nn.Conv2d(intermediate_channels, out_channels, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def wrap(self, output: Tensor) -> Packet[Tensor]:
        return {"segmentation": [output]}

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.conv_3x3(inputs)
        inputs = self.conv_1x1(inputs)
        return self.upscale(inputs)
