## List of supported heads
Every head takes this parameters:
 - n_classes: int # number of classes to predict
 - attach_index: int # on which backbone/neck layer should the head attach to. Defaults to -1.

Here is a list of all supported heads and any additional parameters they take:
- ClassificationHead
  - Params:
    - fc_dropout: float # Dropout rate before last layer, range [0,1]. Defaults to 0.2.
- MultiLabelClassificationHead
  - Params:
    - fc_dropout: float # Dropout rate before last layer, range [0,1]. Defaults to 0.2.
- SegmentationHead (adapted from [here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py))
- BiSeNetHead (adapted from [here](https://github.com/taveraantonio/BiseNetv1))
  - Params:
    - c1: int # Number of input channels. Defaults to 256.
    - upscale_factor: int # Factor used for upscaling input. Defaults to 8.
    - is_aux: bool # Either use 256 for intermediate channels or 64. Defaults to False
- EffiDeHead (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py))
  - Params:
    - n_anchors: int # Should stay default. Defaults to 1.
- YoloV6Head (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/effidehead.py))
  - Params:
    - is_4head: bool # Either build 4 headed architecture or 3 headed one (**Important: Should be same also on backbone and neck**). Defaults to False.
- IKeypoint (adapted from [here](https://github.com/WongKinYiu/yolov7))
  - Params:
    - n_keypoints: int # Number of keypoints
    - anchors: list # Anchors used for object detection
    - connectivity: list # Connectivity mapping used in visualization. Defaults to None.