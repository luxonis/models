## Nodes

Every node takes these parameters:
 - attach_index: int # Index of previous output that the head attaches to. Each node has a sensible default. Usually should not be manually set.

### List
- **ResNet18** ([source](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html))
  - Params:
    - download_weights: bool # If True download weights from imagenet. Defaults to False.

- **MicroNet** ([source](https://github.com/liyunsheng13/micronet))
  - Params:
    - variant: Literal["M1", "M2", "M3"] # Defaults to 'M1'.

- **RepVGG** ([source](https://github.com/DingXiaoH/RepVGG))
  - Params:
    - variant: Literal["A0", "A1", "A2"] # Defaults to "A0".

- **EfficientRep** (adapted from [here](https://arxiv.org/pdf/2209.02976.pdf))
  - Params:
    - channels_list: List[int] # List of number of channels for each block. Defaults to [64, 128, 256, 512, 1024].
    - num_repeats: List[int] # List of number of repeats of RepVGGBlock. Defaults to [1, 6, 12, 18, 6].
    - in_channels: int # Number of input channels, should be 3 in most cases . Defaults to 3.
    - depth_mul: int # Depth multiplier. Defaults to 0.33.
    - width_mul: int # Width multiplier. Defaults to 0.25.

- **RexNetV1_lite** ([source](https://github.com/clovaai/rexnet))
  - Params:
    - fix_head_stem: bool # Whether to multiply head stem. Defaults to False.
    - divisible_value: int # Divisor used. Defaults to 8.
    - input_ch: int # tarting channel dimension. Defaults to 16.
    - final_ch: int # Final channel dimension. Defaults to 164.
    - multiplier: float # Channel dimension multiplier. Defaults to 1.0.
    - kernel_conf: str # Kernel sizes encoded as string. Defaults to '333333'.

- **MobileOne** ([source](https://github.com/apple/ml-mobileone))
  - Params:
    - variant: Literal["s0", "s1", "s2", "s3", "s4"] # Defaults to "s0".

- **MobileNetV2** ([source](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html))
  - Params:
    - download_weights: bool # If True download weights from imagenet. Defaults to False.

- **EfficientNet** ([source](https://github.com/rwightman/gen-efficientnet-pytorch))
  - Params:
    - download_weights: bool # If True download weights from imagenet. Defualts to False.

- **ContextSpatial** (adapted from [here](https://github.com/taveraantonio/BiseNetv1))
  - Params:
    - context_backbone: str # Backbone used. Defaults to 'MobileNetV2'.

- **RepPANNeck** (adapted from [here](https://arxiv.org/pdf/2209.02976.pdf))
  - Params:
    - num_heads: Literal[2,3,4] # Number of output heads. Defaults to 3. ***Note:** Should be same also on head in most cases*.
    - channels_list: List[int] # List of number of channels for each block. Defaults to [256, 128, 128, 256, 256, 512].
    - num_repeats: List[int] # List of number of repeats of RepVGGBlock. Defaults to [12, 12, 12, 12].
    - depth_mul: int # Depth multiplier. Defaults to 0.33.
    - width_mul: int # Width multiplier. Defaults to 0.25.

- **ClassificationHead**
  - Params:
    - fc_dropout: float # Dropout rate before last layer, range [0,1]. Defaults to 0.2.

- **SegmentationHead** (adapted from [here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py))

- **BiSeNetHead** (adapted from [here](https://github.com/taveraantonio/BiseNetv1))
  - Params:
    - upscale_factor: int # Factor used for upscaling input. Defaults to 8.
    - is_aux: bool # Either use 256 for intermediate channels or 64. Defaults to False.

- **EfficientBBoxHead** (adapted from [here](https://arxiv.org/pdf/2209.02976.pdf))
  - Params:
    - num_heads: bool # Number of output heads. Defaults to 3.

- **ImplicitKeypointBBoxHead** (adapted from [here](https://arxiv.org/pdf/2207.02696.pdf))
  - Params:
    - n_keypoints: int # Number of keypoints.
    - num_heads: bool # Number of output heads. Defaults to 3.
    - anchors: List[List[int]] # Anchors used for object detection. Defaults to [ [12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401] ]. *(from COCO)* ***Note:** If this is set to null in config then anchors are computed at runtime from the dataset.*
    - init_coco_biases: bool # Whether to use COCO bias and weight initialization. Defaults to True.
    - conf_thres: float # confidence threshold for nms (used for evaluation)
    - out_thres: float # iou threshold for nms (used for evaluation)
