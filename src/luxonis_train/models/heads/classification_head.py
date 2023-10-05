import torch.nn as nn

from luxonis_train.models.heads.base_heads import BaseClassificationHead
from luxonis_train.utils.registry import HEADS


@HEADS.register_module()
class ClassificationHead(BaseClassificationHead):
    def __init__(
        self,
        n_classes: int,
        input_channels_shapes: list,
        original_in_shape: list,
        attach_index: int = -1,
        fc_dropout: float = 0.2,
        **kwargs
    ):
        """Simple classification head

        Args:
            n_classes (int): Number of classes
            input_channels_shapes (list): List of output shapes from previous module
            original_in_shape (list): Original input shape to the model
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            fc_dropout (float, optional): Dropout rate before last layer, range [0,1]. Defaults to 0.2.
        """
        super().__init__(
            n_classes=n_classes,
            input_channels_shapes=input_channels_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
            **kwargs
        )

        in_channels = self.input_channels_shapes[self.attach_index][1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(in_channels, n_classes),
        )

    def forward(self, x):
        out = self.head(x[self.attach_index])
        return out
