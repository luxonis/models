## List of supported necks
- RepPANNeck (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py))
  - Params:
    - channels_list: List[int] # list of number of channels for each block
    - num_repeats: List[int] # list of number of repeats of RepBlock
    - depth_mul: int # depth multiplier
    - width_mul: int # width multiplier
    - is_4head: bool # either build 4 headed architecture or 3 headed one (**Important: Should be same also on backbone and head**)