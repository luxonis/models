## Heads

Every head takes these parameters:
 - n_classes: int # Number of classes
 - attach_index: int # Index of previous output that the head attaches to. Defaults to -1.
 - main_metric: str # Name of the main metric which is used for tracking training process. Defaults to head specific value.

### List
- **ClassificationHead**
  - Params:
    - fc_dropout: float # Dropout rate before last layer, range [0,1]. Defaults to 0.2.

- **MultiLabelClassificationHead**
  - Params:
    - fc_dropout: float # Dropout rate before last layer, range [0,1]. Defaults to 0.2.

- **SegmentationHead** (adapted from [here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py))

- **BiSeNetHead** (adapted from [here](https://github.com/taveraantonio/BiseNetv1))
  - Params:
    - upscale_factor: int # Factor used for upscaling input. Defaults to 8.
    - is_aux: bool # Either use 256 for intermediate channels or 64. Defaults to False.

- **YoloV6Head** (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py))
  - Params:
    - num_heads: bool # Number of output heads. Defaults to 3. ***Note:** Should be same also on neck in most cases.*
    
  ***Note:** attach_index: Defaults to 0. Value must be non-negative.* 

- **IKeypoint** (adapted from [here](https://github.com/WongKinYiu/yolov7))
  - Params:
    - n_keypoints: int # Number of keypoints
    - num_heads: bool # Number of output heads. Defaults to 3. ***Note:** Should be same also on neck in most cases.*
    - anchors: list # Anchors used for object detection. Defaults to [ [12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401] ]. *(from COCO)*
    - connectivity: list # Connectivity mapping used in visualization. Defaults to None.
    - visibility_threshold: float # Keypoints with visibility lower than threshold won't be drawn. Defaults to 0.5.

  ***Note:** attach_index: Defaults to 0. Value must be non-negative.*