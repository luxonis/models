import torch
import torch.nn as nn
from luxonis_train.models.modules import Up
from luxonis_train.utils.head_type import *

# Note: this is basic FCN Head, doesn't ensure that output is same size as input
# Source: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py

class SegmentationHead(nn.Module):
    def __init__(self, prev_out_shape, n_classes, **kwargs):
        super(SegmentationHead, self).__init__()

        self.n_classes = n_classes
        self.type = SemanticSegmentation()
        original_in_shape = kwargs["original_in_shape"]

        num_up = self._get_num_heads(prev_out_shape[-1][2], original_in_shape[2])
        modules = []
        in_channels = prev_out_shape[-1][1]
        for _ in range(num_up):
            modules.append(Up(in_channels=in_channels, out_channels=in_channels//2))
            in_channels //= 2
        
        self.head = nn.Sequential(
            *modules,
            nn.Conv2d(in_channels, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        out = self.head(x[-1])
        return out


    def _get_num_heads(self, in_dim, out_dim):
        counter = 0
        curr_dim = in_dim
        while curr_dim < out_dim:
            curr_dim *= 2
            counter += 1
        
        if curr_dim != out_dim:
            print("Segmentation head's output shape not same as original input shape")
        return counter 
