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
        self.original_in_shape = kwargs["original_in_shape"]
        self.attach_index = kwargs.get("attach_index", -1)
        self.prev_out_shape = prev_out_shape[self.attach_index]

        num_up = self._get_num_heads(self.prev_out_shape[2], self.original_in_shape[2])
        modules = []
        in_channels = self.prev_out_shape[1]
        for _ in range(num_up):
            modules.append(Up(in_channels=in_channels, out_channels=in_channels//2))
            in_channels //= 2
        
        self.head = nn.Sequential(
            *modules,
            nn.Conv2d(in_channels, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        out = self.head(x[self.attach_index])
        return out


    def _get_num_heads(self, in_height, out_height):
        counter = 0
        curr_height = in_height
        while curr_height < out_height:
            curr_height *= 2
            counter += 1
        
        if curr_height != out_height:
            print("Segmentation head's output shape not same as original input shape")
        return counter 
