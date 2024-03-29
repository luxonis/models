## List of supported loss functions
- CrossEntropyLoss ([source](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html))
  - Params: Can be seen in the source
- BCEWithLogitsLoss ([source](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html))
  - Params: Can be seen in the source
- FocalLoss ([source](https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook))
  - Params:
    - alpha: float
    - gamma: float
- YoloV6Loss (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/loss.py))
  - Params:
    - n_classes: int # should be same as head
    - iou_type: str # giou, diou, ciou or siou
    - loss_weight: dict
    - others should stay default 