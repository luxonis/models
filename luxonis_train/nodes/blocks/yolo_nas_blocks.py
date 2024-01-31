import torch
from torch import nn, Tensor

from collections import OrderedDict
from typing import Type, Union, Mapping, Any, Optional, Iterable, Tuple, List, Callable
import math
from functools import partial

from luxonis_train.utils.nas_utils import (
    generate_anchors_for_grid_cell,
    batch_distance2bbox,
    width_multiplier,
    YoloNasPoseDecodedPredictions,
    YoloNasPoseRawOutputs
)


class Multiply(nn.Module):
  def __init__(self, multiplier=1.0, is_trainable=False):
    super(Multiply, self).__init__()

    if is_trainable:
        self.multiplier = nn.Parameter(torch.tensor([multiplier]), requires_grad=True)
    else:
        self.multiplier = multiplier

  def forward(self, x):
    return self.multiplier * x


class Identity(nn.Module):
  def __init__(self, is_residual=True):
    super(Identity, self).__init__()
    
    self.is_residual = is_residual

  def forward(self, x):
    return x if self.is_residual else 0
  

class QARepVGGBlock(nn.Module):
    """
    Source: Make RepVGG Greater Again: A Quantization-aware Approach (https://arxiv.org/pdf/2212.01593.pdf)
        
                    ┌--- Identity ----------------┐
                    |                             |
                    |                             |
        input ------├─-- 1x1 + bias --- *alpha ---┤--- BN --- ReLU --- output
                    |                             |
                    |                             |
                    └--- 3x3 --- BN --------------┘
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        is_trainable_alpha: bool = False,
        is_residual: bool = False,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param is_trainable_alpha: use alpha as nn.Parameter
        :param is_residual: use residual branch for final output
        """

        super().__init__()

        self.identity_branch = Identity(
            is_residual=is_residual
        )
        self.conv_1x1_branch = nn.Sequential(
            OrderedDict([
                ("conv_1x1", nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=True,
                    dilation=dilation,
                )),
                ("alpha", Multiply(
                    multiplier=1.0, 
                    is_trainable=is_trainable_alpha
                ))
            ])
        )

        self.conv_3x3_branch = nn.Sequential(
            OrderedDict([
                ("conv_3x3", nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )),
                ("BN", nn.BatchNorm2d(
                    num_features=out_channels
                ))
            ])
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):

        conv_3x3 = self.conv_3x3_branch(x)
        conv_1x1 = self.conv_1x1_branch(x)
        residual = self.identity_branch(x)

        output = conv_1x1 + conv_3x3 + residual
        output = self.bn(output)
        output = self.relu(output)

        return output


class ConvBN(nn.Module):
   
    def __init__(self, input_channels, output_channels, kernel, stride, padding=None, groups=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel, stride, self._same_padding(kernel, padding), groups=groups, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)
    
    def _same_padding(self, kernel, padding):
        if padding is None:
            padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
        return padding


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Intended usage of this block is the following:

    >>> class ResNetBlock(nn.Module):
    >>>   def __init__(self, ..., drop_path_rate:float):
    >>>     self.drop_path = DropPath(drop_path_rate)
    >>>
    >>>   def forward(self, x):
    >>>     return x + self.drop_path(self.conv_bn_act(x))

    Code taken from TIMM (https://github.com/rwightman/pytorch-image-models)
    Apache License 2.0
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """

        :param drop_prob: Probability of zeroing out individual vector (channel dimension) of each feature map
        :param scale_by_keep: Whether to scale the output by the keep probability. Enable by default and helps to
                              keep output mean & std in the same range as w/o drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        return self.drop_path(x, self.drop_prob, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
    
    def drop_path(self, x, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        """

        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
    

class YoloNASBottleneck(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        is_residual: bool,
        is_trainable_alpha: bool,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.is_residual = is_residual and input_channels == output_channels
        self.bottleneck = nn.Sequential(
            OrderedDict([
                ("block_1", QARepVGGBlock(input_channels, output_channels, stride=1)),
                ("block_2", QARepVGGBlock(output_channels, output_channels, stride=1)),
                ("drop", DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity())
            ])
        )
        self.residual = nn.Sequential(
            OrderedDict([
                ("residual", Identity(is_residual=self.is_residual)),
                ("alpha", Multiply(multiplier=1.0, is_trainable=is_trainable_alpha)),
            ])
        )

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        residual = self.residual(x)
        output = bottleneck + residual
        return output
    
 
class YoloNASCSPLayer(nn.Module):
    """
    Cross-stage layer module for YoloNAS.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        block_type: Type[nn.Module],
        activation_type: Type[nn.Module],
        shortcut: bool = True,
        use_alpha: bool = True,
        expansion: float = 0.5,
        hidden_channels: int = None,
        concat_intermediates: bool = False,
        drop_path_rates: Union[Iterable[float], None] = None,
        dropout_rate: float = 0.0,
    ):
        """

        :param in_channels: Number of input channels.
        :param out_channels:  Number of output channels.
        :param num_bottlenecks: Number of bottleneck blocks.
        :param block_type: Bottleneck block type.
        :param activation_type: Activation type for all blocks.
        :param shortcut: If True, adds the residual connection from input to output.
        :param use_alpha: If True, adds the learnable alpha parameter (multiplier for the residual connection).
        :param expansion: If hidden_channels is None, hidden_channels is set to in_channels * expansion.
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks.
        :param concat_intermediates:
        :param drop_path_rates: List of drop path probabilities for each bottleneck block.
                                Must have the length equal to the num_bottlenecks or None.
        :param dropout_rate: Dropout probability before the last convolution in this layer.
        """
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_bottlenecks
        else:
            drop_path_rates = tuple(drop_path_rates)
        if len(drop_path_rates) != num_bottlenecks:
            raise ValueError(
                f"Argument drop_path_rates ({drop_path_rates}, len {len(drop_path_rates)} "
                f"must have the length equal to the num_bottlenecks ({num_bottlenecks})."
            )

        super(YoloNASCSPLayer, self).__init__()
        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBN(in_channels, hidden_channels, kernel=1, stride=1)
        self.conv2 = ConvBN(in_channels, hidden_channels, kernel=1, stride=1)
        self.conv3 = ConvBN(hidden_channels * (2 + concat_intermediates * num_bottlenecks), out_channels, kernel=1, stride=1)
        self.bottleneck_modules = [
            YoloNASBottleneck(hidden_channels, hidden_channels, is_residual=shortcut, is_trainable_alpha=use_alpha, drop_path_rate=drop_path_rates[i])
            for i in range(num_bottlenecks)
        ]
        self.bottleneck_modules = nn.Sequential(*self.bottleneck_modules)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottleneck_modules(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.dropout(x)
        return self.conv3(x)


class YoloNASStage(nn.Module):
    """
    A single stage module for YoloNAS. It consists of a downsample block (QARepVGGBlock) followed by YoloNASCSPLayer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation_type: Type[nn.Module] = nn.ReLU,
        hidden_channels: int = None,
        concat_intermediates: bool = False,
        drop_path_rates: Union[Iterable[float], None] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize the YoloNASStage module
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_blocks: Number of bottleneck blocks in the YoloNASCSPLayer
        :param activation_type: Activation type for all blocks
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks.
        :param concat_intermediates: If True, concatenates the intermediate values from the YoloNASCSPLayer.
        :param drop_path_rates: List of drop path probabilities for each bottleneck block.
                                Must have the length equal to the num_blocks or None.
        :param dropout_rate: Dropout probability before the last convolution in this layer.
        """
        # super().__init__(in_channels)
        super().__init__()
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(in_channels, out_channels, is_residual=False, stride=2)
        self.blocks = YoloNASCSPLayer(
            out_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            True,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.blocks(self.downsample(x))


class SPP(nn.Module): # a module for SpatialPyramidPooling
    
    def __init__(self, in_channels, out_channels, k: Tuple):
        super().__init__()

        hidden_channels = in_channels // 2
        self.cv1 = ConvBN(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBN(hidden_channels * (len(k) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class YoloNASPANNeckWithC2(nn.Module):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    where the up-sampling stages include a higher resolution skip
    Returns outputs of neck stage 2, stage 3, stage 4
    """

    def __init__(
        self,
        in_channels: List[int]= [96, 192, 384, 768]
    ):
        """
        Initialize the PAN neck

        :param in_channels: Input channels of the 4 feature maps from the backbone
        :param neck1: First neck stage config
        :param neck2: Second neck stage config
        :param neck3: Third neck stage config
        :param neck4: Fourth neck stage config
        """
        super().__init__()
        c2_out_channels, c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        self.neck1 = YoloNASUpStage(in_channels=[c5_out_channels, c4_out_channels, c3_out_channels], out_channels=192, num_blocks=2, hidden_channels=64, width_mult=1, depth_mult=1, reduce_channels=True, activation_type=nn.ReLU)
        self.neck2 = YoloNASUpStage(in_channels=[self.neck1.out_channels[1], c3_out_channels, c2_out_channels], out_channels=96, num_blocks=2, hidden_channels=48, width_mult=1, depth_mult=1, reduce_channels=True, activation_type=nn.ReLU)
        self.neck3 = YoloNASDownStage(in_channels=[self.neck2.out_channels[1], self.neck2.out_channels[0]], out_channels=192, num_blocks=2, hidden_channels=64, width_mult=1, depth_mult=1, activation_type=nn.ReLU)
        self.neck4 = YoloNASDownStage(in_channels=[self.neck3.out_channels, self.neck1.out_channels[0]], out_channels=384, num_blocks=2, hidden_channels=64, width_mult=1, depth_mult=1, activation_type=nn.ReLU)

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5


class YoloNASUpStage(nn.Module):
    """
    Upsampling stage for YoloNAS.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
        reduce_channels: bool = False,
        drop_path_rates: Union[Iterable[float], None] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize the YoloNASUpStage module
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param width_mult: Multiplier for the number of channels in the stage.
        :param num_blocks: Number of bottleneck blocks
        :param depth_mult: Multiplier for the number of blocks in the stage.
        :param activation_type: Activation type for all blocks
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks
        :param concat_intermediates:
        :param reduce_channels:
        """
        super().__init__()

        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_channels, skip_in_channels = in_channels
        else:
            in_channels, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels  # skip2 downsample results in out_channels channels

        out_channels = math.ceil(int(out_channels * width_mult) / 8) * 8 # width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        if num_inputs == 2:
            self.reduce_skip = ConvBN(skip_in_channels, out_channels, 1, 1) if reduce_channels else nn.Identity()
        else:
            self.reduce_skip1 = ConvBN(skip_in_channels1, out_channels, 1, 1) if reduce_channels else nn.Identity()
            self.reduce_skip2 = ConvBN(skip_in_channels2, out_channels, 1, 1) if reduce_channels else nn.Identity()

        self.conv = ConvBN(in_channels, out_channels, 1, 1)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        if num_inputs == 3:
            self.downsample = ConvBN(out_channels if reduce_channels else skip_in_channels2, out_channels, kernel=3, stride=2)

        self.reduce_after_concat = ConvBN(num_inputs * out_channels, out_channels, 1, 1) if reduce_channels else nn.Identity()

        after_concat_channels = out_channels if reduce_channels else out_channels + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )
        
        self._out_channels = [out_channels, out_channels]
    
    @property
    def out_channels(self):
        return self._out_channels
    
    def forward(self, inputs):
        if len(inputs) == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1, skip_x2 = self.reduce_skip1(skip_x1), self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]
        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], 1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x


class YoloNASDownStage(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
        drop_path_rates: Union[Iterable[float], None] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initializes a YoloNASDownStage.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param width_mult: Multiplier for the number of channels in the stage.
        :param num_blocks: Number of blocks in the stage.
        :param depth_mult: Multiplier for the number of blocks in the stage.
        :param activation_type: Type of activation to use inside the blocks.
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks.
        :param concat_intermediates:
        """

        super().__init__()

        in_channels, skip_in_channels = in_channels
        out_channels = math.ceil(int(out_channels * width_mult) / 8) * 8
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        self.conv = ConvBN(in_channels, out_channels // 2, 3, 2)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            in_channels=after_concat_channels,
            out_channels=out_channels,
            num_bottlenecks=num_blocks,
            block_type=partial(ConvBN, kernel=3, stride=1),
            activation_type=activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )

        self._out_channels = out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], 1)
        x = self.blocks(x)
        return x


class YoloNASPoseNDFLHeads(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        # heads_list: List[Union[HpmStruct, DictConfig]],
        blocks_config: List[dict],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        inference_mode: bool = False,
        eval_size: Optional[Tuple[int, int]] = None,
        width_mult: float = 1.0,
        pose_offset_multiplier: float = 1.0,
        compensate_grid_cell_offset: bool = True,
    ):
        """
        Initializes the NDFLHeads module.

        :param num_classes: Number of detection classes
        :param in_channels: Number of channels for each feature map (See width_mult)
        :param grid_cell_scale: A scaling factor applied to the grid cell coordinates.
               This scaling factor is used to define anchor boxes (see generate_anchors_for_grid_cell).
        :param grid_cell_offset: A fixed offset that is added to the grid cell coordinates.
               This offset represents a 'center' of the cell and is 0.5 by default.
        :param reg_max: Number of bins in the regression head
        :param eval_size: (rows, cols) Size of the image for evaluation. Setting this value can be beneficial for inference speed,
               since anchors will not be regenerated for each forward call.
        :param width_mult: A scaling factor applied to in_channels.
        :param pose_offset_multiplier: A scaling factor applied to the pose regression offset. This multiplier is
               meant to reduce absolute magnitude of weights in pose regression layers.
               Default value is 1.0.
        :param compensate_grid_cell_offset: (bool) Controls whether to subtract anchor cell offset from the pose regression.
               If True, predicted pose coordinates decoded as (offsets + anchors - grid_cell_offset) * stride.
               If False, predicted pose coordinates decoded as (offsets + anchors) * stride.
               Default value is True.

        """
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        super().__init__()

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size
        self.pose_offset_multiplier = pose_offset_multiplier
        self.compensate_grid_cell_offset = compensate_grid_cell_offset
        self.inference_mode = inference_mode

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        self._init_weights()

        # factory = det_factory.DetectionModulesFactory()
        # heads_list = self._insert_heads_list_params(heads_list, factory, num_classes, reg_max)

        self.num_heads = len(blocks_config)
        fpn_strides: List[int] = []
        for i in range(self.num_heads):
            block_config = blocks_config[i]
            new_head = YoloNASPoseDFLHead(**block_config)
            fpn_strides.append(new_head.stride)
            setattr(self, f"head{i + 1}", new_head)

        self.fpn_strides = tuple(fpn_strides)

    @torch.jit.ignore
    def _init_weights(self):
        if self.eval_size:

            device, dtype = None, None

            try:
                device = next(iter(self.parameters())).device
            except StopIteration:
            
                device =  next(iter(self.buffers())).device

            try:
                dtype = next(iter(self.parameters())).dtype
            except:
                dtype = next(iter(self.buffers())).dtype

            # device = infer_model_device(self)
            # dtype = infer_model_dtype(self)

            anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats: Tuple[Tensor, ...]) -> Union[YoloNasPoseDecodedPredictions, Tuple[YoloNasPoseDecodedPredictions, YoloNasPoseRawOutputs]]:
        """
        Runs the forward for all the underlying heads and concatenate the predictions to a single result.
        :param feats: List of feature maps from the neck of different strides
        :return: Return value depends on the mode:
        If tracing, a tuple of 4 tensors (decoded predictions) is returned:
        - pred_bboxes [B, Num Anchors, 4] - Predicted boxes in XYXY format
        - pred_scores [B, Num Anchors, 1] - Predicted scores for each box
        - pred_pose_coords [B, Num Anchors, Num Keypoints, 2] - Predicted poses in XY format
        - pred_pose_scores [B, Num Anchors, Num Keypoints] - Predicted scores for each keypoint

        In training/eval mode, a tuple of 2 tensors returned:
        - decoded predictions - they are the same as in tracing mode
        - raw outputs - a tuple of 8 elements in total, this is needed for training the model.
        """

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        pose_regression_list = []
        pose_logits_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit, pose_regression, pose_logits = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            pose_regression_list.append(torch.permute(pose_regression.flatten(3), [0, 3, 1, 2]))  # [B, J, 2, H, W] -> [B, H * W, J, 2]
            pose_logits_list.append(torch.permute(pose_logits.flatten(2), [0, 2, 1]))  # [B, J, H, W] -> [B, H * W, J]

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, J, 2]
        pose_logits_list = torch.cat(pose_logits_list, dim=1)  # [B, Anchors, J]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        # Decode keypoints
        if self.pose_offset_multiplier != 1.0:
            pose_regression_list *= self.pose_offset_multiplier

        if self.compensate_grid_cell_offset:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2) - self.grid_cell_offset
        else:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2)

        pose_regression_list *= stride_tensor.unsqueeze(0).unsqueeze(2)

        pred_pose_coords = pose_regression_list.detach().clone()  # [B, Anchors, C, 2]
        pred_pose_scores = pose_logits_list.detach().clone().sigmoid()  # [B, Anchors, C]

        decoded_predictions = pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores

        if torch.jit.is_tracing() or self.inference_mode:
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = cls_score_list, reg_distri_list, pose_regression_list, pose_logits_list, anchors, anchor_points, num_anchors_list, stride_tensor
        return decoded_predictions, raw_predictions

    @property
    def out_channels(self):
        return None

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []

        dtype = dtype or feats[0].dtype
        device = device or feats[0].device

        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)

        if device is not None:
            anchor_points = anchor_points.to(device)
            stride_tensor = stride_tensor.to(device)
        return anchor_points, stride_tensor


class YoloNASPoseDFLHead(nn.Module):
    """
    YoloNASPoseDFLHead is the head used in YoloNASPose model.
    This class implements single-class object detection and keypoints regression on a single scale feature map
    """

    def __init__(
        self,
        in_channels: int,
        bbox_inter_channels: int,
        pose_inter_channels: int,
        pose_regression_blocks: int,
        shared_stem: bool,
        pose_conf_in_class_head: bool,
        pose_block_use_repvgg: bool,
        width_mult: float,
        first_conv_group_size: int,
        num_classes: int,
        stride: int,
        reg_max: int,
        cls_dropout_rate: float = 0.0,
        reg_dropout_rate: float = 0.0,
    ):
        """
        Initialize the YoloNASDFLHead
        :param in_channels: Input channels
        :param bbox_inter_channels: Intermediate number of channels for box detection & regression
        :param pose_inter_channels: Intermediate number of channels for pose regression
        :param shared_stem: Whether to share the stem between the pose and bbox heads
        :param pose_conf_in_class_head: Whether to include the pose confidence in the classification head
        :param width_mult: Width multiplier
        :param first_conv_group_size: Group size
        :param num_classes: Number of keypoints classes for pose regression. Number of detection classes is always 1.
        :param stride: Output stride for this head
        :param reg_max: Number of bins in the regression head
        :param cls_dropout_rate: Dropout rate for the classification head
        :param reg_dropout_rate: Dropout rate for the regression head
        """
        super().__init__()

        bbox_inter_channels = width_multiplier(bbox_inter_channels, width_mult, 8)
        pose_inter_channels = width_multiplier(pose_inter_channels, width_mult, 8)

        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = bbox_inter_channels // first_conv_group_size

        self.num_classes = num_classes
        self.shared_stem = shared_stem
        self.pose_conf_in_class_head = pose_conf_in_class_head

        if self.shared_stem:
            max_input = max(bbox_inter_channels, pose_inter_channels)
            self.stem = ConvBNReLU(in_channels, max_input, kernel_size=1, stride=1, padding=0, bias=False)

            if max_input != pose_inter_channels:
                self.pose_stem = nn.Conv2d(max_input, pose_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.pose_stem = nn.Identity()

            if max_input != bbox_inter_channels:
                self.bbox_stem = nn.Conv2d(max_input, bbox_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.bbox_stem = nn.Identity()

        else:
            self.stem = nn.Identity()
            self.pose_stem = ConvBNReLU(in_channels, pose_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bbox_stem = ConvBNReLU(in_channels, bbox_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        if pose_block_use_repvgg:
            pose_block = partial(QARepVGGBlock, use_alpha=True)
        else:
            pose_block = partial(ConvBNReLU, kernel_size=3, stride=1, padding=1, bias=False)

        pose_convs = [pose_block(pose_inter_channels, pose_inter_channels) for _ in range(pose_regression_blocks)]
        self.pose_convs = nn.Sequential(*pose_convs)

        self.reg_pred = nn.Conv2d(bbox_inter_channels, 4 * (reg_max + 1), 1, 1, 0)

        if self.pose_conf_in_class_head:
            self.cls_pred = nn.Conv2d(bbox_inter_channels, 1 + self.num_classes, 1, 1, 0)
            self.pose_pred = nn.Conv2d(pose_inter_channels, 2 * self.num_classes, 1, 1, 0)  # each keypoint is x,y
        else:
            self.cls_pred = nn.Conv2d(bbox_inter_channels, 1, 1, 1, 0)
            self.pose_pred = nn.Conv2d(pose_inter_channels, 3 * self.num_classes, 1, 1, 0)  # each keypoint is x,y,confidence

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.stride = stride

        self.prior_prob = 1e-2
        self._initialize_biases()

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        if self.pose_conf_in_class_head:
            self.cls_pred = compute_new_weights_fn(self.cls_pred, 1 + num_classes)
            self.pose_pred = compute_new_weights_fn(self.pose_pred, 2 * num_classes)
        else:
            self.pose_pred = compute_new_weights_fn(self.pose_pred, 3 * num_classes)
        self.num_classes = num_classes

    @property
    def out_channels(self):
        return None

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        :param x: Input feature map of shape [B, Cin, H, W]
        :return: Tuple of [reg_output, cls_output, pose_regression, pose_logits]
            - reg_output:      Tensor of [B, 4 * (reg_max + 1), H, W]
            - cls_output:      Tensor of [B, 1, H, W]
            - pose_regression: Tensor of [B, num_classes, 2, H, W]
            - pose_logits:     Tensor of [B, num_classes, H, W]
        """
        x = self.stem(x)
        pose_features = self.pose_stem(x)
        bbox_features = self.bbox_stem(x)

        cls_feat = self.cls_convs(bbox_features)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(bbox_features)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        pose_feat = self.pose_convs(pose_features)
        pose_feat = self.reg_dropout_rate(pose_feat)

        pose_output = self.pose_pred(pose_feat)

        if self.pose_conf_in_class_head:
            pose_logits = cls_output[:, 1:, :, :]
            cls_output = cls_output[:, 0:1, :, :]
            pose_regression = pose_output.reshape((pose_output.size(0), self.num_classes, 2, pose_output.size(2), pose_output.size(3)))
        else:
            pose_output = pose_output.reshape((pose_output.size(0), self.num_classes, 3, pose_output.size(2), pose_output.size(3)))
            pose_logits = pose_output[:, :, 2, :, :]
            pose_regression = pose_output[:, :, 0:2, :, :]

        return reg_output, cls_output, pose_regression, pose_logits

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)


class ConvBNReLU(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Activation layer.
        Default behaviour is Conv-BN-Act. To exclude Batchnorm module use
        `use_normalization=False`, to exclude activation use `activation_type=None`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    """

    def __init__(
       self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        use_normalization: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        use_activation: bool = True,
        inplace: bool = False,
    ):

        super().__init__()

        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
        )

        if use_normalization:
            self.seq.add_module(
                "bn",
                nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype),
            )
        if use_activation is not None:
            self.seq.add_module("act", nn.ReLU(dict(inplace=inplace)))

    def forward(self, x):
        return self.seq(x)

    def get_input_channels(self) -> int:
        return self.seq[0].in_channels
