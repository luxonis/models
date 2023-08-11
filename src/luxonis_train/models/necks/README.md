## List of supported necks
- RepPANNeck (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py))
  - Params:
    - channels_list: List[int] # List of number of channels for each block
    - num_repeats: List[int] # List of number of repeats of RepVGGBlock
    - depth_mul: int # Depth multiplier. Defaults to 0.33.
    - width_mul: int # Width multiplier. Defaults to 0.25.
    - is_4head: bool # Either build 4 headed architecture or 3 headed one (**Important: Should be same also on backbone and head**). Defaults to False.