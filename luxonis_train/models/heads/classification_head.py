import torch
import torch.nn as nn
from luxonis_train.utils.head_type import *

class ClassificationHead(nn.Module):
    def __init__(self, prev_out_shape, n_classes=0, n_labels=0, fc_dropout=0.2, **kwargs):
        super(ClassificationHead, self).__init__()

        assert (n_classes or n_labels), "Classification head should have at least one class/label"
        assert ((n_classes>0 and n_labels==0) or (n_classes==0 and n_labels>0)), "Head should be either multi-class or multi-label. Consider using 2 heads"

        self.n_classes = n_classes
        self.n_labels = n_labels
        self.type = Classification() if self.n_classes > 0 else MultiLabelClassification()
        self.original_in_shape = kwargs["original_in_shape"]
        self.prev_out_shape = prev_out_shape
        
        self.in_channels = self.prev_out_shape[-1][1]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(self.in_channels, n_classes if isinstance(self.type, Classification) else n_labels)
        )

    def forward(self, x):
        out = self.head(x[-1])
        return out
