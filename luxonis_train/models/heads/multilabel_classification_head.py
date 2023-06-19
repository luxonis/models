import torch
import torch.nn as nn
from luxonis_train.utils.head_type import *

class MultiLabelClassificationHead(nn.Module):
    def __init__(self, prev_out_shape, n_classes=0, fc_dropout=0.2, **kwargs):
        super(MultiLabelClassificationHead, self).__init__()

        self.n_classes = n_classes
        self.type = MultiLabelClassification()
        self.original_in_shape = kwargs["original_in_shape"]
        self.attach_index = kwargs.get("attach_index", -1)
        self.prev_out_shape = prev_out_shape[self.attach_index]
        
        self.in_channels = self.prev_out_shape[1]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(self.in_channels, n_classes)
        )

    def forward(self, x):
        out = self.head(x[self.attach_index])
        return out
