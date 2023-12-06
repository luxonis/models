from torch import Tensor, nn

from .base_node import BaseNode


class ClassificationHead(BaseNode[Tensor, Tensor]):
    in_channels: int

    def __init__(
        self,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """Simple classification head.

        Args:
            dropout_rate (float, optional): Dropout rate before last layer, range [0,1].
              Defaults to 0.2.
        """
        # TODO: fix attach index
        super().__init__(attach_index=-1, **kwargs)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_channels, self.dataset_metadata.n_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(inputs)
