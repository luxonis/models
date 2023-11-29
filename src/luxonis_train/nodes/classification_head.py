from torch import Tensor, nn

from .base_node import BaseNode


class ClassificationHead(BaseNode[Tensor, Tensor]):
    in_channels: int

    def __init__(
        self,
        n_classes: int | None = None,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """Simple classification head.

        Args:
            n_classes (int): Number of classes
            dropout_rate (float, optional): Dropout rate before last layer, range [0,1].
              Defaults to 0.2.
            attach_index (int, optional): Index of previous output that this
              head attaches to. Defaults to -1.
        """
        super().__init__(attach_index=-1, **kwargs)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_channels, n_classes or self.dataset_metadata.n_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(inputs)
