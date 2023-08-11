## List of supported necks
- RepPANNeck (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py))
  - Params:
    - num_heads (Literal[2,3,4]): Number of output heads.
      (**Important: Should be same also on the head**). Defaults to 3.
    - channels_list: List[int] # List of number of channels for each block. Defaults to [256, 128, 128, 256, 256, 512].
    - num_repeats: List[int] # List of number of repeats of RepBlock. Defaults to [12, 12, 12, 12].
    - depth_mul: int # Depth multiplier. Defaults to 0.33.
    - width_mul: int # Width multiplier. Defaults to 0.25.