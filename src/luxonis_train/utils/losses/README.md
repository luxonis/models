## Losses

Every loss takes these parameters:
- head_attributes: Dict[str, Any], optional # Dictionary of all head attributes to which the loss is connected to. Defaults to {}.

## List
- **CrossEntropyLoss** ([source](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html))
  - Params: Can be seen in the source

- **BCEWithLogitsLoss** ([source](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html))
  - Params: Can be seen in the source

- **SigmoidFocalLoss** ([source](https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html#torchvision.ops.sigmoid_focal_loss))
  - Params:
    - alpha: float # Defaults to 0.8.
    - gamma: float # Defaults to 2.0.
    - reduction: Literal["none", "mean", "sum"] # Defaults to "mean".

- **SoftmaxFocalLoss**:
  - Params:
    - alpha: Union[float, list] # Either a float for all channels or list of alphas for each channel with length C. Defaults to 0.25.
    - gamma: float # Defaults to 2.0.
    - reduction: Literal["none", "mean", "sum"] # Defaults to "mean".

- **BboxYoloV6Loss** (adapted from [here](https://arxiv.org/pdf/2209.02976.pdf))
  - Params:
    - n_warmup_epochs: int # Number of epochs where ATSS assigner is used, after that we switch to TAL assigner. Defaults to 4.
    - iou_type: Literal["none", "giou", "diou", "ciou", "siou"] # IoU type used for bbox regression loss. Defaults to "giou".
    - loss_weight: Dict[str, float] # Mapping for sub losses weights. Defautls to {"class": 1.0, "iou": 2.5}.