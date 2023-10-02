## Necks

Every neck takes these parameters:
 - attach_index: int # Index of previous output that the neck attaches to. Defaults to -1.


### List
- **RepPANNeck** (adapted from [here](https://arxiv.org/pdf/2209.02976.pdf))
  - Params:
    - num_heads: Literal[2,3,4] # Number of output heads. Defaults to 3. ***Note:** Should be same also on head in most cases*.
    - channels_list: List[int] # List of number of channels for each block. Defaults to [256, 128, 128, 256, 256, 512].
    - num_repeats: List[int] # List of number of repeats of RepVGGBlock. Defaults to [12, 12, 12, 12].
    - depth_mul: int # Depth multiplier. Defaults to 0.33.
    - width_mul: int # Width multiplier. Defaults to 0.25.

  ***Note:** attach_index: Defaults to -1. Value must be negative.* 