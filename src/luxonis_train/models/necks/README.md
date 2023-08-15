## List of supported necks
- RepPANNeck (adapted from [here](https://github.com/meituan/YOLOv6/blob/725913050e15a31cd091dfd7795a1891b0524d35/yolov6/models/reppan.py))
  - Params:
    - num_heads: Literal[2,3,4] # Number of output heads.
      (**Important: Should be same also on the head**). Defaults to 3.
    - offset: int # Offset used if want to use backbone's higher resolution outputs.
      If num_heads==2 then this can be one of [0,1,2], if num_heads==3 then this can be one [1,2], if num_heads==4 then this must be 0. (**Important: Should be same also on head**). Defaults to 0.
    - channels_list: List[int] # List of number of channels for each block. Defaults to [256, 128, 128, 256, 256, 512].
    - num_repeats: List[int] # List of number of repeats of RepVGGBlock. Defaults to [12, 12, 12, 12].
    - depth_mul: int # Depth multiplier. Defaults to 0.33.
    - width_mul: int # Width multiplier. Defaults to 0.25.