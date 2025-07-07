from timm.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from einops import repeat
from timm.layers import DropPath

from collections.abc import Sequence

from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He

import collections
import inspect
import itertools
import math
import warnings
from functools import partial
from itertools import repeat
from typing import Optional, Dict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from monai.networks.blocks import Convolution
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import UpSample
from monai.utils import UpsampleMode, InterpolateMode
# from prettytable import PrettyTable
from torch import Tensor
from torch.nn import functional as F

CONV_MODELS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,
}


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `conv_layer` cannot be found
    # in the registry, fallback to search `conv_layer` in the
    # mmengine.MODELS.
    conv_layer = CONV_MODELS[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".

    Example:
        # >>> kernel_size = 16
        # >>> stride = 16
        # >>> dilation = 1
        # >>> input = torch.rand(1, 1, 15, 17)
        # >>> adap_pad = AdaptivePadding(
        # >>>     kernel_size=kernel_size,
        # >>>     stride=stride,
        # >>>     dilation=dilation,
        # >>>     padding="corner")
        # >>> out = adap_pad(input)
        # >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        # >>> input = torch.rand(1, 1, 16, 17)
        # >>> out = adap_pad(input)
        # >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Add padding to `x`

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


def get_dwconv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        padding: int | tuple[int, ...] = None,

):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels,
                             padding=padding)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=1, kernel_size=1, bias=bias, conv_only=True, groups=1, padding=padding)
    return torch.nn.Sequential(depth_conv, point_conv)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 spatial_dims: int,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation: int | tuple[int, ...] = 1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None):
        super().__init__()
        padding = padding[:spatial_dims]
        dilation = dilation[:spatial_dims]
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        # kernel_size = to_2tuple(kernel_size)
        # stride = to_2tuple(stride)
        # dilation = to_2tuple(dilation)
        #
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None

        self.projection = get_dwconv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        try:
            x = self.projection(x)
        except:
            x = F.pad(x, (0, 0, 0, 0, 0, 1, 0, 0, 0, 0))
            x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float,
                           b: float) -> Tensor:
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor,
                  mean: float = 0.,
                  std: float = 1.,
                  a: float = -2.,
                  b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


NORM_MODELS = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    # 'SyncBN': SyncBatchNorm,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
}


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        norm_layer = NORM_MODELS[layer_type]
    abbr = str(layer_type)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def resize_pos_embed(pos_embed: torch.Tensor,
                     src_shape: Tuple[int],
                     dst_shape: Tuple[int],
                     mode: str = 'trilinear',
                     num_extra_tokens: int = 1) -> torch.Tensor:
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (T, H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (T, H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'trilinear'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1] \
            and src_shape[2] == dst_shape[2]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_t, src_h, src_w = src_shape
    assert L == src_t * src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_t}*{src_h}*{src_w}+{num_extra_tokens}).' \
        'Please check the `img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_t, src_h, src_w,
                                    C).permute(0, 4, 1, 2, 3)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``, We rename the ``p`` of
    ``torch.nn.Dropout`` to ``drop_prob`` so as to be consistent with
    ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.
    """

    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


class Block(nn.Module):
    def __init__(
            self, spatial_dims: int, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, reverse=False,
            drop_path_rate=0.0, drop_rate=0.0
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        # self.split_head = split_head
        self.reverse = reverse
        # self.transpose = transpose
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.dropout = Dropout(drop_prob=drop_rate)

        # if use_mlp:
        #     self.ffn = SwiGLUFFNFused(
        #         embed_dims=dim,
        #         feedforward_channels=int(dim * 4),
        #         layer_scale_init_value=0.0)
        #     self.ln2 = build_norm_layer(dict(type='LN'), dim)
        # else:
        #     self.ffn = None
        self.ffn = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, order='t l h w',
            shape=None, skip=True, n_dim_pos=4
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        # h = w = 0
        assert shape is not None
        if self.spatial_dims == 3:
            t, h, w = shape
        else:
            h, w = shape
            t = 1
        if n_dim_pos != 4:
            order = order.split(' ')
            assert len(order) == 4
            trunc_n = 4 - n_dim_pos
            tgt_order = f"(n {' '.join(order[:trunc_n])}) ({' '.join(order[trunc_n:])}) c"
        else:
            tgt_order = f'n ({order}) c'
        hidden_states = rearrange(hidden_states, f'n (t h w ) c -> {tgt_order}', t=t, h=h, w=w)
        if self.reverse:
            hidden_states = hidden_states.flip(1)
            if residual is not None:
                residual = residual.flip(1)
        if not self.fused_add_norm:
            hidden_states = self.norm(hidden_states)
            if skip:
                hidden_states = hidden_states + self.drop_path(
                    self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
            else:
                hidden_states = self.drop_path(
                    self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states = self.drop_path(self.mixer(hidden_states, inference_params=inference_params))
        # if self.ffn is not None:
        #     hidden_states = self.ffn(self.ln2(hidden_states), identity=hidden_states)
        if self.reverse:
            hidden_states = hidden_states.flip(1)
            # if residual is not None:
            #     residual = residual.flip(1)
        hidden_states = rearrange(hidden_states, f'{tgt_order}->n (t h w ) c ', t=t, h=h, w=w)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        spatial_dims: int,
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        reverse=None,
        drop_rate=0.1,
        drop_path_rate=0.1):
    ssm_cfg = ssm_cfg or dict()
    factory_kwargs = {"device": device, "dtype": dtype}
    # if use_nd:
    #     transpose = False
    #     reverse = False
    # mixer_cls = partial(UNETR, layer_idx=layer_idx, n_dim=n_dim, **ssm_cfg, **factory_kwargs)
    # mixer_cls = partial(Mamba2D if is_2d else Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # if is_2d:
    #     block = Block2D(
    #         d_model,
    #         mixer_cls,
    #         norm_cls=norm_cls,
    #         fused_add_norm=fused_add_norm,
    #         residual_in_fp32=residual_in_fp32,
    #         reverse=reverse,
    #         drop_rate=drop_rate,
    #         transpose=transpose,
    #         drop_path_rate=drop_path_rate,
    #     )
    # else:
    block = Block(
        spatial_dims,
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        reverse=reverse,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    block.layer_idx = layer_idx
    return block


def get_scales(spatial_dims, input_patch_size, n_layers, patch_size):
    if input_patch_size is not None:
        scales = []
        if spatial_dims == 3:
            z, h, w = input_patch_size
            if patch_size is not None:
                z_p, h_p, w_p = patch_size if not isinstance(patch_size, int) else (patch_size, patch_size, patch_size)

                z, h, w = z / z_p, h / h_p, w / w_p

            for step in range(n_layers):
                z_scale, z = get_scale(z)
                h_scale, h = get_scale(h)
                w_scale, w = get_scale(w)
                scales.append((z_scale, h_scale, w_scale))
        elif spatial_dims == 2:
            h, w = input_patch_size
            if patch_size is not None:
                h_p, w_p = patch_size if not isinstance(patch_size, int) else (patch_size, patch_size)

                h, w = h / h_p, w / w_p

            for step in range(n_layers):
                h_scale, h = get_scale(h)
                w_scale, w = get_scale(w)
                scales.append((h_scale, w_scale))
    else:
        scales = None
    return scales


def permute(x, spatial_dims: int, reverse=False):
    if spatial_dims == 2:
        if reverse:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
    elif spatial_dims == 3:
        if reverse:
            x = x.permute(0, 4, 1, 2, 3)
        else:
            x = x.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError
    return x


def get_scale_value(spatial_dims: int, input_patch_size, scales):
    if spatial_dims == 3:
        z, h, w = input_patch_size
        for z_p, h_p, w_p in scales:
            z, h, w = z / z_p, h / h_p, w / w_p
        return z, h, w
    elif spatial_dims == 2:
        h, w = input_patch_size
        for h_p, w_p in scales:
            h, w = h / h_p, w / w_p
        return h, w
    else:
        raise Exception()


def shape(x, spatial_dims, channel_first=True):
    if spatial_dims == 2:
        Z = None
        if channel_first:
            B, C, H, W = x.shape
            return B, C, Z, H, W
        else:
            B, H, W, C = x.shape
            return B, Z, H, W, C
    elif spatial_dims == 3:
        if channel_first:
            B, C, Z, H, W = x.shape
            return B, C, Z, H, W
        else:
            B, Z, H, W, C = x.shape
            return B, Z, H, W, C
    else:
        raise Exception()


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer. The output will have scale times features and H and W will be divided by scale times.
    Args:
        input_dim (int): Resolution of input feature.
        scale (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, spatial_dims: int, input_dim: int, scale: Union[int, Tuple[int, ...]],
                 output_features: int,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.spatial_dims = spatial_dims

        if self.spatial_dims == 2:
            self.hs, self.ws = (scale, scale) if isinstance(scale, int) else scale[:spatial_dims]
            self.zs = 1
        elif self.spatial_dims == 3:
            self.zs, self.hs, self.ws = (scale, scale, scale) if isinstance(scale, int) else scale[:spatial_dims]

        self.input_feature_size = (self.zs * self.ws * self.hs) * input_dim
        self.output_features = output_features
        self.reduction = nn.Linear(self.input_feature_size, self.output_features, bias=False)
        self.norm = norm_layer(self.input_feature_size)

    @staticmethod
    def _patch_merge(spatial_dims: int, x: torch.tensor, zs: int, hs: int, ws: int):
        if spatial_dims == 3:
            outputs = [[0] if zs == 1 else [0, 1],
                       [0] if hs == 1 else [0, 1],
                       [0] if ws == 1 else [0, 1],
                       ]
        else:
            outputs = [[0] if hs == 1 else [0, 1],
                       [0] if ws == 1 else [0, 1],
                       ]
        x_outputs = []
        outputs = itertools.product(*outputs)
        for comb in outputs:
            if spatial_dims == 3:
                xi = x[:, comb[0]::zs, comb[1]::hs, comb[2]::ws, :]
            else:
                xi = x[:, comb[0]::hs, comb[1]::ws, :]
            x_outputs.append(xi)

        x = torch.cat([t for t in x_outputs if np.prod(t.shape) != 0], -1)
        return x

    def forward(self, x, permute_=False):
        if permute_:
            x = permute(x, self.spatial_dims).contiguous()
        B, Z, H, W, C = shape(x, spatial_dims=self.spatial_dims, channel_first=False)

        pad_input = (H % self.hs == 1) or (W % self.ws == 1) or (Z and (Z % self.zs == 1))
        if pad_input:
            if self.spatial_dims == 3:
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, Z % 2))
            elif self.spatial_dims == 2:
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2,))

        B, Z, H, W, C = shape(x, spatial_dims=self.spatial_dims, channel_first=False)
        SHAPE_FIX = [-1, -1]
        if (W % self.ws != 0) or (H % self.hs != 0) or ((Z or 2) % self.zs != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            if self.spatial_dims == 2:
                SHAPE_FIX[0] = H // self.scale
                SHAPE_FIX[1] = W // self.scale
            elif self.spatial_dims == 3:
                SHAPE_FIX[0] = Z // self.scale
                SHAPE_FIX[1] = H // self.scale
                SHAPE_FIX[2] = W // self.scale
        # if self.spatial_dims == 2:
        # x0 = x[:, 0::self.hs, 0::self.ws, :]  # B H/self.scale W/self.scale C
        # x1 = x[:, 1::self.hs, 0::self.ws, :]  # B H/self.scale W/self.scale C
        # x2 = x[:, 0::self.hs, 1::self.ws, :]  # B H/self.scale W/self.scale C
        # x3 = x[:, 1::self.hs, 1::self.ws, :]  # B H/self.scale W/self.scale C
        # x = torch.cat([t for t in [x0, x1, x2, x3] if np.prod(t.shape) != 0], -1)  # B H/2 W/2 4*C
        # x = self._patch_merge3d(x, self.hs, self.ws)
        # elif self.spatial_dims == 3:
        x = self._patch_merge(self.spatial_dims, x, self.zs, self.hs, self.ws)

        if self.spatial_dims == 2:
            x = x.view(B, H // self.hs, W // self.ws, (self.ws * self.hs) * C)  # B H/2*W/2 4*C
        else:
            x = x.view(B, Z // self.zs, H // self.hs, W // self.ws,
                       (self.zs * self.hs * self.ws) * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        if permute_:
            x = permute(x, self.spatial_dims, reverse=True).contiguous()
        return x


def get_scale(scale_value, scale_factor=2):
    if scale_value % scale_factor == 1:
        f_scale = 1
    else:
        f_scale = scale_factor
    return f_scale, scale_value // f_scale


def _upsample_like(src, tar, upsample_mode: UpsampleMode | str = "nontrainable"):
    # src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    # get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
    layer = UpSample(
        spatial_dims=len(src.shape[2:]),
        in_channels=src.shape[1],
        out_channels=src.shape[1],
        size=tar.shape[2:],
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )
    return layer(src)


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, spatial_dims: int, dim: int, scale: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]],
                 output_dim: int = None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.dim = dim
        self.output_dim = output_dim
        scale = scale if isinstance(scale, int) else scale[:spatial_dims]
        if isinstance(scale, int):
            self.zs = self.hs = self.ws = self.cs = scale
            self.zs = self.zs if spatial_dims == 3 else 1
        else:
            if len(scale) == 4 and spatial_dims == 3:
                self.zs, self.hs, self.ws, self.cs = scale
            elif len(scale) == 3 and spatial_dims == 3:
                self.zs, self.hs, self.ws = scale
                self.cs = None
            elif len(scale) == 3 and spatial_dims == 2:
                self.hs, self.ws, self.cs = scale
                self.zs = 1
            elif len(scale) == 2 and spatial_dims == 2:
                self.hs, self.ws = scale
                self.cs = None
                self.zs = 1
            else:
                raise Exception()
            if self.output_dim is not None and self.cs is not None:
                raise ValueError("output_dim and cs cannot be not None at the same time!")

        if self.output_dim is None:
            output_dim = self.zs * self.hs * self.ws // self.cs
            self.expand = nn.Linear(dim, output_dim, bias=False)
            self.norm = norm_layer(dim // self.cs)
        else:
            self.expand = nn.Linear(dim // (self.zs * self.hs * self.ws), self.output_dim, bias=False)
            self.norm = norm_layer(self.output_dim)

    def forward(self, x, permute_=False):

        if self.output_dim is None:
            x = permute(x, self.spatial_dims)  # B, C, H, W ==> B, H, W, C
            x = self.expand(x)

            B, Z, H, W, C = shape(x, channel_first=False, spatial_dims=self.spatial_dims)
            # x = x.view(B, H, W, C)
            if self.spatial_dims == 2:
                x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.hs, p2=self.ws,
                              c=C // (self.hs * self.ws))
            elif self.spatial_dims == 3:
                x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c', p1=self.zs, p2=self.hs,
                              p3=self.ws, c=C // (self.zs * self.hs * self.ws))
            else:
                raise Exception()

            x = x.view(B, -1, C // (self.zs * self.hs * self.ws))
            x = self.norm(x)

            if self.spatial_dims == 2:
                x = x.reshape(B, H * self.hs, W * self.ws, C // (self.hs * self.ws))
            elif self.spatial_dims == 3:
                x = x.reshape(B, Z * self.zs, H * self.hs, W * self.ws, C // (self.zs * self.hs * self.ws))
            else:
                raise Exception()
        else:
            x = permute(x, self.spatial_dims)  # B, C, H, W ==> B, H, W, C
            B, Z, H, W, C = shape(x, channel_first=False, spatial_dims=self.spatial_dims)

            # x = x.view(B, H, W, C)
            if self.spatial_dims == 2:
                x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                              p1=self.hs,
                              p2=self.ws,
                              c=C // (self.hs * self.ws))
            elif self.spatial_dims == 3:
                x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c',
                              p1=self.zs,
                              p2=self.hs,
                              p3=self.ws, c=C // (self.zs * self.hs * self.ws))
            else:
                raise Exception()

            x = self.expand(x)  # new shape!
            x = x.view(B, -1, self.output_dim)
            x = self.norm(x)
            if self.spatial_dims == 2:
                x = x.reshape(B, H * self.hs, W * self.ws, self.output_dim)
            elif self.spatial_dims == 3:
                x = x.reshape(B, Z * self.zs, H * self.hs, W * self.ws, self.output_dim)
            else:
                raise Exception()
        if permute_:
            x = permute(x, self.spatial_dims, reverse=True).contiguous()
        return x


class UNETR2Net(nn.Module):

    def __init__(self,
                 spatial_dims: int,
                 in_channels: int, out_channels: int, deep_supervision: bool,
                 input_patch_size, add_last: bool = True):
        super(UNETR2Net, self).__init__()
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.input_patch_size = input_patch_size
        scales = get_scales(spatial_dims, input_patch_size, n_layers=5, patch_size=None)
        self.scales = scales
        self.stage1 = UNETR(spatial_dims=spatial_dims,
                            in_channels=in_channels,
                            out_channels=32,
                            feature_size=4,
                            hidden_size=96,
                            num_layers=7,
                            patch_size=(16, 16, 16),
                            img_size=input_patch_size,
                            add_last=add_last,
                            # decoder_scale=(2, 2, 2, 2),
                            # encoder_layers=(2, 1, 0),
                            )
        # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.patch_merging1 = PatchMerging2D(spatial_dims, 32, scale=scales[0], output_features=64)
        # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = UNETR(spatial_dims=spatial_dims,
                            in_channels=64,
                            out_channels=64,
                            feature_size=4,
                            hidden_size=192,
                            num_layers=6,
                            patch_size=(16, 16, 16),
                            img_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                            add_last=add_last,
                            # decoder_scale=(2, 2, 2, 1),
                            )
        self.patch_merging2 = PatchMerging2D(spatial_dims, 64, scale=scales[1],
                                             output_features=128)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = UNETR(spatial_dims=spatial_dims,
                            in_channels=128,
                            out_channels=128,
                            feature_size=8,
                            hidden_size=384,
                            num_layers=5,
                            patch_size=(8, 8, 8),
                            img_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                            add_last=add_last,
                            decoder_scale=(2, 2, 2, 1),
                            # encoder_layers=(2, 1, 0),
                            )
        self.patch_merging3 = PatchMerging2D(spatial_dims, 128, scale=scales[2],
                                             output_features=256)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = UNETR(spatial_dims=spatial_dims,
                            in_channels=256,
                            out_channels=256,
                            feature_size=8,
                            hidden_size=384,
                            num_layers=4,
                            patch_size=(4, 4, 4),
                            img_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                            encoder_layers=(1, 1, 0),
                            decoder_scale=(2, 2, 1, 1),
                            add_last=add_last,
                            )
        self.patch_merging4 = PatchMerging2D(spatial_dims, 256, scale=scales[3],
                                             output_features=512)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = UNETR(spatial_dims=spatial_dims,
                            in_channels=512,
                            out_channels=512,
                            feature_size=16,
                            hidden_size=384,
                            num_layers=4,
                            patch_size=(2, 2, 2),
                            encoder_layers=(0, 0, 0),
                            decoder_scale=(2, 1, 1, 1),
                            img_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                            add_last=add_last,
                            )
        self.patch_merging5 = PatchMerging2D(spatial_dims, 512,
                                             scale=(1, 1, 1),
                                             output_features=512)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage6 = UNETR(spatial_dims=spatial_dims,
                            in_channels=512,
                            out_channels=512,
                            feature_size=16,
                            hidden_size=384,
                            num_layers=4,
                            patch_size=(2, 2, 2),
                            encoder_layers=(0, 0, 0),
                            decoder_scale=(2, 1, 1, 1),
                            img_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                            add_last=add_last,
                            )
        # spatial_dims, 512, 256, 512)  # in: 512, 7, 7 -> 512, 7, 7

        # decoder
        self.patch_expand5d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=512,
            scale=(1, 1, 1),
            norm_layer=nn.LayerNorm,
            output_dim=512
        )  # 512, 14, 14 -> 512, 28, 28
        # -> concat -> 512, 28, 28
        self.stage5d = UNETR(spatial_dims=spatial_dims,
                             in_channels=1024,
                             out_channels=512,
                             feature_size=16,
                             hidden_size=384,
                             num_layers=4,
                             patch_size=(2, 2, 2),
                             encoder_layers=(0, 0, 0),
                             decoder_scale=(2, 1, 1, 1),
                             img_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                             add_last=add_last,
                             )
        # spatial_dims, 1024, 256, 512)  # in: 1024, 14, 14 -> 512, 14, 14

        self.patch_expand4d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=512,
            scale=scales[-2],
            norm_layer=nn.LayerNorm,
            output_dim=256
        )  # 512, 14, 14 -> 256, 28, 28
        # -> concat -> 512, 28, 28

        self.concat_back_dim4d = nn.Linear(512, 256)

        self.stage4d = UNETR(spatial_dims=spatial_dims,
                             in_channels=256,
                             out_channels=256,
                             feature_size=8,
                             hidden_size=384,
                             num_layers=4,
                             patch_size=(2, 2, 2),
                             encoder_layers=(0, 0, 0),
                             decoder_scale=(2, 1, 1, 1),
                             img_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                             add_last=add_last,
                             )
        self.patch_expand3d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=256,
            scale=scales[-3],
            norm_layer=nn.LayerNorm,
            output_dim=128,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        self.concat_back_dim3d = nn.Linear(256, 128)  # 128 56 56
        self.stage3d = UNETR(spatial_dims=spatial_dims,
                             in_channels=128,
                             out_channels=128,
                             feature_size=4,
                             hidden_size=384,
                             num_layers=5,
                             patch_size=(4, 4, 4),
                             encoder_layers=(1, 1, 0),
                             decoder_scale=(2, 2, 1, 1),
                             img_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                             add_last=add_last,
                             )
        self.patch_expand2d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=128,
            scale=scales[-4],
            norm_layer=nn.LayerNorm,
            output_dim=64,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        self.concat_back_dim2d = nn.Linear(128, 64)  # 64, 112, 112
        self.stage2d = UNETR(spatial_dims=spatial_dims,
                             in_channels=64,
                             out_channels=64,
                             feature_size=4,
                             hidden_size=192,
                             num_layers=6,
                             patch_size=(8, 8, 8),
                             decoder_scale=(2, 2, 2, 1),
                             img_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                             add_last=add_last,
                             )
        self.patch_expand1d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=64,
            scale=scales[-5],
            norm_layer=nn.LayerNorm,
            output_dim=32
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        self.concat_back_dim1d = nn.Linear(64, 32)  # 64, 112, 112
        self.stage1d = UNETR(spatial_dims=spatial_dims,
                             in_channels=32,
                             out_channels=32,
                             feature_size=4,
                             hidden_size=96,
                             num_layers=7,
                             patch_size=(16, 16, 16),
                             img_size=input_patch_size,
                             add_last=add_last,
                             )

        self.side1 = Convolution(spatial_dims, 32, out_channels, kernel_size=1, padding=0, conv_only=True)
        self.side2 = Convolution(spatial_dims, 64, out_channels, kernel_size=1, padding=0, conv_only=True)
        self.side3 = Convolution(spatial_dims, 128, out_channels, kernel_size=1, padding=0, conv_only=True)
        self.side4 = Convolution(spatial_dims, 256, out_channels, kernel_size=1, padding=0, conv_only=True)
        self.side5 = Convolution(spatial_dims, 512, out_channels, kernel_size=1, padding=0, conv_only=True)
        self.side6 = Convolution(spatial_dims, 512, out_channels, kernel_size=1, padding=0, conv_only=True)

        self.outconv = Convolution(spatial_dims, 6 * out_channels, out_channels, kernel_size=1, conv_only=True)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)  # 32, 224, 224
        hx = self.patch_merging1(hx1, permute_=True)  # 64, 112, 112

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.patch_merging2(hx2, permute_=True)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.patch_merging3(hx3, permute_=True)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.patch_merging4(hx4, permute_=True)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.patch_merging5(hx5, permute_=True)

        # stage 6
        hx6 = self.stage6(hx)

        # -------------------- decoder --------------------
        hx6up = self.patch_expand5d(hx6, permute_=True)  # 1024, 14, 14

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # in: 1024, 14, 14 -> out: 512, 14, 14
        hx5dup = self.patch_expand4d(hx5d)  # 512, 14, 14 -> 256, 28, 28
        hx5dup = permute(self.concat_back_dim4d(torch.cat((hx5dup, permute(hx4, self.spatial_dims)), -1)),
                         self.spatial_dims, reverse=True)  # .permute(0, 3, 1, 2)

        # 512, 14, 14 -> 256, 28, 28
        hx4d = self.stage4d(hx5dup)
        hx4dup = self.patch_expand3d(hx4d)
        hx4dup = permute(self.concat_back_dim3d(torch.cat((hx4dup, permute(hx3, self.spatial_dims)), -1)),
                         self.spatial_dims, reverse=True)
        # 512, 14, 14 -> 256, 28, 28

        hx3d = self.stage3d(hx4dup)
        hx3dup = self.patch_expand2d(hx3d)
        hx3dup = permute(self.concat_back_dim2d(torch.cat((hx3dup, permute(hx2, self.spatial_dims)), -1)),
                         self.spatial_dims, reverse=True)

        # 512, 14, 14 -> 256, 28, 28
        hx2d = self.stage2d(hx3dup)
        hx2dup = self.patch_expand1d(hx2d)
        hx2dup = permute(self.concat_back_dim1d(torch.cat((hx2dup, permute(hx1, self.spatial_dims)), -1)),
                         self.spatial_dims, reverse=True)

        hx1d = self.stage1d(hx2dup)

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        # d2 = _upsample_like(d2, d1.shape[2:])

        d3 = self.side3(hx3d)
        # d3 = _upsample_like(d3, d1.shape[2:])

        d4 = self.side4(hx4d)
        # d4 = _upsample_like(d4, d1.shape[2:])

        d5 = self.side5(hx5d)
        # d5 = _upsample_like(d5, d1.shape[2:])

        d6 = self.side6(hx6)
        # d6 = _upsample_like(d6, d1.shape[2:])

        d0 = self.outconv(torch.cat((d1,
                                     _upsample_like(d2, d1),
                                     _upsample_like(d3, d1),
                                     _upsample_like(d4, d1),
                                     _upsample_like(d5, d1),
                                     _upsample_like(d6, d1)),
                                    1))

        # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        if self.deep_supervision:
            return d0, d1, d2, d3, d4, d5, d6
        else:
            return d0

    @torch.no_grad()
    def freeze_encoder(self):
        for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.patch_merging1,
                      self.patch_merging2, self.patch_merging3, self.patch_merging4, self.pool56]:
            for name, param in group.named_parameters():
                # if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.patch_merging1,
                      self.patch_merging2, self.patch_merging3, self.patch_merging4, self.pool56]:
            for param in group.parameters():
                param.requires_grad = True


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            img_size: Sequence[int] | int,
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            proj_type: str = "conv",
            norm_name: tuple | str = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,

            qkv_bias: bool = False,
            save_attn: bool = False,
            num_layers: int = 7,
            patch_size: tuple[int, ...] = (16, 16, 16),
            decoder_scale: tuple[int, ...] = (2, 2, 2, 2),
            encoder_scale: tuple[int, ...] = (2, 2, 2),
            encoder_layers: tuple[int, ...] = (2, 1, 0),
            add_last: bool = True
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
        """

        super().__init__()
        self.add_last = add_last
        if self.add_last:
            self.rebnconvin = get_dwconv_layer(2, in_channels, out_channels)
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.spatial_dims = spatial_dims
        self.num_layers = num_layers
        img_size = tuple(int(item) for item in ensure_tuple_rep(img_size, spatial_dims))
        self.patch_size = patch_size[:spatial_dims]
        # self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.feat_size = self.get_feat_size(spatial_dims, img_size, patch_size)
        self.hidden_size = hidden_size
        self.classification = False
        self.out_indices = [int(item) for item in np.linspace(2, num_layers - 1, 3)]
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,

        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=encoder_layers[0],
            kernel_size=3,
            stride=1,
            upsample_kernel_size=encoder_scale[2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,

        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=encoder_layers[1],
            kernel_size=3,
            stride=1,
            upsample_kernel_size=encoder_scale[1],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=encoder_layers[2],
            kernel_size=3,
            stride=1,
            upsample_kernel_size=encoder_scale[0],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=decoder_scale[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=decoder_scale[1],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=decoder_scale[2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=decoder_scale[3],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    @staticmethod
    def get_feat_size(spatial_dims: int, img_size, patch_size):
        if spatial_dims == 3:
            return (
                int(img_size[0] // patch_size[0]),
                int(img_size[1] // patch_size[1]),
                int(img_size[2] // patch_size[2]),
            )
        elif spatial_dims == 2:
            return (
                int(img_size[0] // patch_size[0]),
                int(img_size[1] // patch_size[1]),
            )

    @staticmethod
    def proj_feat(spatial_dims: int, x, hidden_size, feat_size):
        x = x.view(x.size(0), *feat_size, hidden_size)
        x = permute(x, spatial_dims=spatial_dims, reverse=True).contiguous()
        # x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    # def proj_feat(self, x):
    #     new_view = [x.size(0)] + self.proj_view_shape
    #     x = x.view(new_view)
    #     x = x.permute(self.proj_axes).contiguous()
    #     return x

    def forward(self, x_in):
        if self.add_last:
            last_add = self.rebnconvin(x_in)

        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[self.out_indices[0]]
        enc2 = self.encoder2(self.proj_feat(self.spatial_dims, x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[self.out_indices[1]]
        enc3 = self.encoder3(self.proj_feat(self.spatial_dims, x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[self.out_indices[2]]
        enc4 = self.encoder4(self.proj_feat(self.spatial_dims, x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(self.spatial_dims, x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        out = self.out(out)
        if self.add_last:
            out = out + last_add
        return out


def get_unetr2net_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        use_pretrain: bool = True,
        small_mode: bool = False
):
    # dim = len(configuration_manager.conv_kernel_sizes[0])
    # assert dim == 2, "Only 2D supported at the moment"
    label_manager = plans_manager.get_label_manager(dataset_json)
    if not small_mode:
        model = UNETR2Net(
            spatial_dims=len(configuration_manager.patch_size),
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            deep_supervision=deep_supervision,
            input_patch_size=configuration_manager.patch_size
        )

    else:
        raise NotImplementedError()
        # model = UNETR2NetP(
        #     spatial_dims=len(configuration_manager.patch_size),
        #     factorization_type="cross-scan",
        #     in_channels=num_input_channels,
        #     out_channels=label_manager.num_segmentation_heads,
        #     deep_supervision=deep_supervision,
        #     input_patch_size=configuration_manager.patch_size
        # )

    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


if __name__ == '__main__':
    # s = (4, 12, 12)
    # model = UNETR(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=3,
    #     img_size=s,
    #     feature_size=4,
    #     hidden_size=96,
    #     norm_name="instance",
    #     conv_block=True,
    #     res_block=True,
    #     dropout_rate=0.0,
    #     num_layers=4,
    #     patch_size=(2, 2, 2),
    #     encoder_scale=(2, 2, 2),
    #     decoder_scale=(2, 1, 1, 1),
    #     encoder_layers=(0, 0, 0)
    # ).to("cuda:0")
    # output = model(torch.randn(size=(1, 1, *s)).to("cuda:0"))
    # print(output.shape)
    from torchinfo import summary

    #
    # print(summary(model, input_size=[1, 1, *s]))

    in_channels = 1
    out_channels = 1
    input_patch_size_ = [192, 192]
    model = UNETR2Net(spatial_dims=len(input_patch_size_),
                      # factorization_type="cross-scan",
                      in_channels=in_channels,
                      out_channels=out_channels,
                      deep_supervision=True,
                      input_patch_size=input_patch_size_).to("cuda")
    output = model(torch.rand(size=(1, in_channels, *input_patch_size_)).to("cuda"))
    for x in output:
        print(x.shape)

    print(summary(model, input_size=[1, in_channels, *input_patch_size_]))
