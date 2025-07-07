import numpy as np
import itertools
from typing import Union, Tuple

from mamba_ssm.modules.mamba2 import Mamba2 as Mamba
# from mamba_ssm import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import InterpolateMode
from einops import rearrange
from monai.networks.blocks.upsample import UpSample
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from monai.utils import UpsampleMode
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Dropout
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer


def get_dwconv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


class UnetDecoderUpSampleLayer(nn.Module):
    def __init__(self, spatial_dims: int, in_ch: int):
        super().__init__()
        self.spatial_up_sample = get_upsample_layer(spatial_dims=spatial_dims, in_channels=in_ch)
        self.feature_down_sample = get_dwconv_layer(spatial_dims=spatial_dims, in_channels=in_ch,
                                                    out_channels=in_ch // 2)

    def forward(self, enc_x, x):
        x = self.spatial_up_sample(x)
        x = self.feature_down_sample(x)
        return torch.cat([enc_x, x], dim=1)


class MambaLayer(nn.Module):
    @staticmethod
    def get_nheaddim(d_model, expand):
        nheaddim = 1
        for i in range(1, int((d_model * expand / 8))):
            if d_model * expand / i % 8 == 0:
                nheaddim = i
        return nheaddim

    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            **(dict(headdim=self.get_nheaddim(input_dim, expand)) if Mamba.__name__ == "Mamba2" else dict())
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class MaxPool(nn.Module):
    def __init__(self, spatial_dims: int,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices: bool = False,
                 ceil_mode: bool = False):
        super().__init__()
        if spatial_dims == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation,
                                         return_indices=return_indices,
                                         ceil_mode=ceil_mode)
        elif spatial_dims == 3:
            self.max_pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation,
                                         return_indices=return_indices,
                                         ceil_mode=ceil_mode)

    def forward(self, input):
        return self.max_pool(input)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


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


class InstanceNorm(nn.Module):
    def __init__(self, spatial_dims: int,
                 in_channels: int):
        super().__init__()
        if spatial_dims == 2:
            self.layer = nn.InstanceNorm2d(in_channels)
        elif spatial_dims == 3:
            self.layer = nn.InstanceNorm3d(in_channels)

    def forward(self, input):
        return self.layer(input)


class GSC(nn.Module):
    def __init__(self, spatial_dims: int, in_channels) -> None:
        super().__init__()

        self.proj = get_dwconv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels,
                                     stride=1, bias=True)
        # Convolution(spatial_dims, in_ch, in_ch, kernel_size=3, strides=1, padding=1,
        # conv_only=True)
        self.norm = InstanceNorm(spatial_dims, in_channels)
        self.nonliner = nn.ReLU()

        self.proj2 = Convolution(spatial_dims, in_channels, in_channels,
                                 kernel_size=1, strides=1, padding=0,
                                 conv_only=True)
        self.norm2 = InstanceNorm(spatial_dims, in_channels)
        self.nonliner2 = nn.ReLU()

        self.proj3 = get_dwconv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels,
                                      stride=1, bias=True)
        self.norm3 = InstanceNorm(spatial_dims, in_channels)
        self.nonliner3 = nn.ReLU()


    def forward(self, x):
        x_residual = x

        x1 = self.norm(x)
        x1 = self.proj(x1)
        x1 = self.nonliner(x1)

        x2 = self.norm2(x)
        x2 = self.proj2(x2)
        x2 = self.nonliner2(x2)

        x = x1 + x2
        x3 = self.norm3(x)
        x3 = self.proj3(x3)
        x3 = self.nonliner3(x3)

        return x3 + x_residual


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
            self.hs, self.ws = (scale, scale) if isinstance(scale, int) else scale
            self.zs = 1
        elif self.spatial_dims == 3:
            self.zs, self.hs, self.ws = (scale, scale, scale) if isinstance(scale, int) else scale

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


class ResUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("SiLU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)

        return x


class ResMambaBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
            order: str = "d h w"
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.order = order
        self.spatial_dims = spatial_dims
        self.gsc = GSC(spatial_dims, in_channels)
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.mamba1 = MambaLayer(input_dim=in_channels, output_dim=in_channels)
        self.mamba2 = MambaLayer(input_dim=in_channels, output_dim=in_channels)

    def forward(self, x):
        """
        In which order to create the mamba
        :param x:
        :return:
        """

        x = self.gsc(x)
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.nd_mamba_order(self.order, x, self.mamba1)
        x = self.norm2(x)
        x = self.act(x)
        x = self.nd_mamba_order(self.order, x, self.mamba2)

        x += identity

        return x

    def nd_mamba_order(self, order: str, x: torch.Tensor, mamba_module: nn.Module):
        if self.spatial_dims == 3:
            b, c, d, h, w = x.shape
            org_order = "b c d h w"
            tgt_order = f'b c {order}'
            x = rearrange(x, f'{org_order} -> {tgt_order}', d=d, h=h, w=w)
            x = mamba_module(x)
            x = rearrange(x, f'{tgt_order}-> b c d h w', d=d, h=h, w=w)
        else:
            b, c, h, w = x.shape
            org_order = "b c h w"
            tgt_order = f'b c {order}'
            x = rearrange(x, f'{org_order} -> {tgt_order}', h=h, w=w)
            x = mamba_module(x)
            x = rearrange(x, f'{tgt_order}-> b c h w', h=h, w=w)
        return x


def get_scale(scale_value, scale_factor=2):
    if scale_value % scale_factor == 1:
        f_scale = 1
    else:
        f_scale = scale_factor
    return f_scale, scale_value // f_scale


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


def get_scales(spatial_dims, input_patch_size, n_layers, patch_size=None, min_size: int = 1):
    # if min_size > 1:
    #     n_layers = n_layers - int(np.log2(min_size))

    if input_patch_size is not None:
        scales = []
        if spatial_dims == 3:
            z, h, w = input_patch_size
            if patch_size is not None:
                z_p, h_p, w_p = patch_size if not isinstance(patch_size, int) else (patch_size, patch_size, patch_size)

                z, h, w = z / z_p, h / h_p, w / w_p

            for step in range(n_layers):
                z_scale, z_ = get_scale(z)
                h_scale, h_ = get_scale(h)
                w_scale, w_ = get_scale(w)
                z, z_scale = (z_, z_scale) if z_ >= min_size else (z, 1)
                h, h_scale = (h_, h_scale) if h_ >= min_size else (h, 1)
                w, w_scale = (w_, w_scale) if w_ >= min_size else (w, 1)

                scales.append((z_scale, h_scale, w_scale))
        elif spatial_dims == 2:
            h, w = input_patch_size
            if patch_size is not None:
                h_p, w_p = patch_size if not isinstance(patch_size, int) else (patch_size, patch_size)

                h, w = h / h_p, w / w_p

            for step in range(n_layers):
                h_scale, h_ = get_scale(h)
                w_scale, w_ = get_scale(w)
                h, h_scale = (h_, h_scale) if h_ >= min_size else (h, 1)
                w, w_scale = (w_, w_scale) if w_ >= min_size else (w, 1)
                scales.append((h_scale, w_scale))
    else:
        scales = None
    # if min_size > 1:
    #     scales += [tuple([1] * spatial_dims)] * int(np.log2(min_size))

    return scales


class LightMUNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 3,
            mid_ch: int = 32,
            in_ch: int = 1,
            out_ch: int = 2,
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            n_layers: int = 7,
            # skip_last_downsample: bool = False,
            add_last: bool = False,
            # blocks_down: tuple = (1, 2, 2, 4),
            # blocks_up: tuple = (1, 1, 1),
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
            min_size: int = 4,
            input_patch_size: tuple[int, ...] = None
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.input_path_size = input_patch_size
        self.add_last = add_last
        # self.skip_last_downsample = skip_last_downsample
        self.spatial_dims = spatial_dims
        self.init_filters = mid_ch
        self.in_channels = in_ch
        self.n_layers = n_layers
        self.layer_in_channels = [mid_ch] * n_layers
        self.blocks_down = [1] + [2] * (n_layers - 1)  # + [4] Lower the number of mamba :)
        self.blocks_up = [1] * (n_layers - 1)
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        self.scales = [(1, 1, 1)[:spatial_dims]] + get_scales(spatial_dims, input_patch_size, n_layers - 1,
                                                              min_size=min_size)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_ch, mid_ch)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_ch)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _get_down_samples(self):
        n_scales = 0
        for _ in range(self.n_layers):
            for x in input_path_size:
                if x < self.min_size:
                    break
            input_path_size = np.array(self.input_path_size) / 2
        return n_scales

    def _make_down_layers(self):
        if self.spatial_dims == 3:
            orders = (
                'd h w',
                'd w h',
                'w h d'
            )
        else:
            orders = (
                'h w',
                'w h',
            )
        down_layers = nn.ModuleList()
        for i, item in enumerate(self.blocks_down):
            # layer_in_channels = filters * 2 ** i
            layer_in_channels = self.layer_in_channels[i]
            # _mamba_layer = MambaLayer(input_dim=layer_in_channels, output_dim=layer_in_channels)
            downsample = MaxPool(self.spatial_dims, kernel_size=self.scales[i], stride=self.scales[i]) if np.prod(
                self.scales[i]) != 1 else nn.Identity()
            down_layer = nn.Sequential(
                # _mamba_layer,
                downsample,
                *[ResMambaBlock(self.spatial_dims, layer_in_channels, norm=self.norm, act=self.act,
                                order=orders[i % len(orders)]) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            # sample_in_channels = filters * 2 ** (n_up - i)
            sample_in_channels = self.layer_in_channels[i]
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels, kernel_size=1),
                    get_upsample_layer(spatial_dims,
                                       sample_in_channels,
                                       upsample_mode=upsample_mode,
                                       scale_factor=self.scales[-(i + 1)]) if np.prod(
                        self.scales[-(i + 1)]) != 1 else nn.Identity()

                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up_sample, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up_sample(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        if self.add_last:
            # change the shape and add it :)
            pass
        return x

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


##### M^2-Net ####
class LightMamba2Net(nn.Module):

    def __init__(self, spatial_dims: int, in_ch: int,
                 out_ch: int, deep_supervision: bool, input_patch_size):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_patch_size = input_patch_size
        scales = get_scales(spatial_dims, input_patch_size, n_layers=5, patch_size=None, min_size=8)
        self.scales = scales
        self.deep_supervision = deep_supervision
        self.stage1 = LightMUNet(spatial_dims=spatial_dims,
                                 in_ch=in_ch,
                                 mid_ch=16,
                                 out_ch=32,
                                 n_layers=7,
                                 input_patch_size=input_patch_size,
                                 add_last=True)
        self.patch_merging1 = PatchMerging2D(spatial_dims, 32, scale=scales[0],
                                             output_features=64)  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = LightMUNet(spatial_dims=spatial_dims, in_ch=64, mid_ch=32, out_ch=64, n_layers=6,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                                 add_last=True)
        self.patch_merging2 = PatchMerging2D(spatial_dims, 64, scale=scales[1],
                                             output_features=128)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = LightMUNet(spatial_dims=spatial_dims, in_ch=128, mid_ch=64, out_ch=128, n_layers=5,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                                 add_last=True)
        self.patch_merging3 = PatchMerging2D(spatial_dims, 128, scale=scales[2],
                                             output_features=256)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = LightMUNet(spatial_dims=spatial_dims, in_ch=256, mid_ch=128, out_ch=256, n_layers=4,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                                 add_last=True)
        self.patch_merging4 = PatchMerging2D(spatial_dims, 256, scale=scales[3],
                                             output_features=512)
        # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = LightMUNet(spatial_dims=spatial_dims, in_ch=512, mid_ch=256, out_ch=512, n_layers=4,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                                 add_last=True)
        self.patch_merging5 = PatchMerging2D(spatial_dims, 512,
                                             scale=scales[4],
                                             output_features=512)
        # in: 512, 14, 14 -> out: 512, 7, 7

        self.stage6 = LightMUNet(spatial_dims, in_ch=512, mid_ch=256, out_ch=512,
                                 n_layers=4,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:5]))
        # in: 512, 7, 7 -> 512, 7, 7

        self.patch_expand5d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=512,
            scale=scales[-1],
            norm_layer=nn.LayerNorm,
            output_dim=512
        )
        # decoder
        self.stage5d = LightMUNet(spatial_dims, in_ch=1024, mid_ch=256, out_ch=512,
                                  n_layers=4,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                                  add_last=True)  # in: 512, 7, 7 -> 512, 7, 7

        self.patch_expand4d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=512,
            scale=scales[-2],
            norm_layer=nn.LayerNorm,
            output_dim=256
        )  # 512, 14, 14 -> 256, 28, 28
        # -> concat -> 512, 28, 28

        self.concat_back_dim4d = nn.Linear(512, 256)

        self.stage4d = LightMUNet(spatial_dims=spatial_dims, in_ch=256, mid_ch=128, out_ch=256, n_layers=4,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                                  add_last=True)
        self.patch_expand3d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=256,
            scale=scales[-3],
            norm_layer=nn.LayerNorm,
            output_dim=128,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        self.concat_back_dim3d = nn.Linear(256, 128)  # 128 56 56
        self.stage3d = LightMUNet(spatial_dims=spatial_dims, in_ch=128, mid_ch=64, out_ch=128, n_layers=5,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                                  add_last=True)  # 128 56 56
        self.patch_expand2d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=128,
            scale=scales[-4],
            norm_layer=nn.LayerNorm,
            output_dim=64,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        self.concat_back_dim2d = nn.Linear(128, 64)  # 64, 112, 112
        self.stage2d = LightMUNet(spatial_dims=spatial_dims, in_ch=64, mid_ch=32, out_ch=64, n_layers=6,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                                  add_last=True)  # 64, 112, 112
        self.patch_expand1d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=64,
            scale=scales[-5],
            norm_layer=nn.LayerNorm,
            output_dim=32
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        self.concat_back_dim1d = nn.Linear(64, 32)  # 64, 112, 112
        self.stage1d = LightMUNet(spatial_dims=spatial_dims, in_ch=32, mid_ch=16, out_ch=32, n_layers=7,
                                  input_patch_size=input_patch_size, add_last=True)

        self.side1 = Convolution(spatial_dims, 32, out_ch, kernel_size=1, padding=0, conv_only=True)
        self.side2 = Convolution(spatial_dims, 64, out_ch, kernel_size=1, padding=0, conv_only=True)
        self.side3 = Convolution(spatial_dims, 128, out_ch, kernel_size=1, padding=0, conv_only=True)
        self.side4 = Convolution(spatial_dims, 256, out_ch, kernel_size=1, padding=0, conv_only=True)
        self.side5 = Convolution(spatial_dims, 512, out_ch, kernel_size=1, padding=0, conv_only=True)
        self.side6 = Convolution(spatial_dims, 512, out_ch, kernel_size=1, padding=0, conv_only=True)

        self.outconv = Convolution(spatial_dims, 6 * out_ch, out_ch, kernel_size=1, conv_only=True)

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
        # hx6up = _upsample_like(hx6, hx5.shape[2:])  # 512, 14, 14

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


class LightMamba2NetP(nn.Module):

    def __init__(self, spatial_dims: int, in_ch: int,
                 out_ch: int, deep_supervision: bool, input_patch_size):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_patch_size = input_patch_size
        scales = get_scales(spatial_dims, input_patch_size, n_layers=5, patch_size=None, min_size=8)
        self.scales = scales
        self.deep_supervision = deep_supervision

        encoder_layer_info = [dict(in_ch=in_ch, mid_ch=32, out_ch=64),
                              dict(in_ch=64, mid_ch=32, out_ch=64),
                              dict(in_ch=64, mid_ch=32, out_ch=64),
                              dict(in_ch=64, mid_ch=32, out_ch=64),
                              dict(in_ch=64, mid_ch=32, out_ch=64),
                              dict(in_ch=64, mid_ch=32, out_ch=64),
                              ]

        patch_merging_info = [
            dict(input_dim=encoder_layer_info[0]['out_ch'], output_features=encoder_layer_info[1]['in_ch']),
            dict(input_dim=encoder_layer_info[1]['out_ch'], output_features=encoder_layer_info[2]['in_ch']),
            dict(input_dim=encoder_layer_info[2]['out_ch'], output_features=encoder_layer_info[3]['in_ch']),
            dict(input_dim=encoder_layer_info[3]['out_ch'], output_features=encoder_layer_info[4]['in_ch']),
            dict(input_dim=encoder_layer_info[4]['out_ch'], output_features=encoder_layer_info[5]['in_ch']),
        ]
        decoder_layer_info = [
            dict(in_ch=128, mid_ch=32, out_ch=128),
            dict(in_ch=128, mid_ch=32, out_ch=128),
            dict(in_ch=128, mid_ch=32, out_ch=128),
            dict(in_ch=128, mid_ch=32, out_ch=128),
            dict(in_ch=128, mid_ch=32, out_ch=128),
        ]
        patch_expansion_info = [
            dict(dim=patch_merging_info[-1]['output_features'], output_dim=patch_merging_info[-1]['output_features']),
            dict(dim=decoder_layer_info[0]["out_ch"], output_dim=decoder_layer_info[0]["out_ch"] // 2),
            dict(dim=decoder_layer_info[1]["out_ch"], output_dim=decoder_layer_info[1]["out_ch"] // 2),
            dict(dim=decoder_layer_info[2]["out_ch"], output_dim=decoder_layer_info[2]["out_ch"] // 2),
            dict(dim=decoder_layer_info[3]["out_ch"], output_dim=decoder_layer_info[3]["out_ch"] // 2),
        ]
        concat_back_info = [
            dict(in_features=patch_expansion_info[1]["output_dim"] * 2,
                 out_features=decoder_layer_info[1]['in_ch']),

            dict(in_features=patch_expansion_info[2]["output_dim"] * 2,
                 out_features=decoder_layer_info[2]['in_ch']),

            dict(in_features=patch_expansion_info[3]["output_dim"] * 2,
                 out_features=decoder_layer_info[3]['in_ch']),

            dict(in_features=patch_expansion_info[4]["output_dim"] * 2,
                 out_features=decoder_layer_info[4]['in_ch']),

        ]
        self.stage1 = LightMUNet(spatial_dims=spatial_dims,
                                 **encoder_layer_info[0],
                                 n_layers=7,
                                 input_patch_size=input_patch_size,
                                 add_last=True)
        self.patch_merging1 = PatchMerging2D(spatial_dims, **patch_merging_info[0],
                                             scale=scales[0])  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = LightMUNet(spatial_dims=spatial_dims, **encoder_layer_info[1], n_layers=6,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                                 add_last=True)
        self.patch_merging2 = PatchMerging2D(spatial_dims, **patch_merging_info[1],
                                             scale=scales[1])  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = LightMUNet(spatial_dims=spatial_dims, **encoder_layer_info[2], n_layers=5,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                                 add_last=True)
        self.patch_merging3 = PatchMerging2D(spatial_dims, **patch_merging_info[2],
                                             scale=scales[2])  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = LightMUNet(spatial_dims=spatial_dims, **encoder_layer_info[3], n_layers=4,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                                 add_last=True)
        self.patch_merging4 = PatchMerging2D(spatial_dims, **patch_merging_info[3], scale=scales[3])
        # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = LightMUNet(spatial_dims=spatial_dims, **encoder_layer_info[4], n_layers=4,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                                 add_last=True)
        self.patch_merging5 = PatchMerging2D(spatial_dims, **patch_merging_info[4],
                                             scale=scales[4])
        # in: 512, 14, 14 -> out: 512, 7, 7

        self.stage6 = LightMUNet(spatial_dims, **encoder_layer_info[5],
                                 n_layers=4,
                                 input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:5]))
        # in: 512, 7, 7 -> 512, 7, 7

        self.patch_expand5d = PatchExpand(
            spatial_dims=spatial_dims,
            **patch_expansion_info[0],
            scale=scales[-1],
            norm_layer=nn.LayerNorm,
        )
        # decoder
        self.stage5d = LightMUNet(spatial_dims, **decoder_layer_info[0],
                                  n_layers=4,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:4]),
                                  add_last=True)  # in: 512, 7, 7 -> 512, 7, 7

        self.patch_expand4d = PatchExpand(
            spatial_dims=spatial_dims,
            **patch_expansion_info[1],
            scale=scales[-2],
            norm_layer=nn.LayerNorm,
        )  # 512, 14, 14 -> 256, 28, 28
        # -> concat -> 512, 28, 28

        self.concat_back_dim4d = self.get_concat_model(**concat_back_info[0])

        self.stage4d = LightMUNet(spatial_dims=spatial_dims, **decoder_layer_info[1], n_layers=4,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                                  add_last=True)
        self.patch_expand3d = PatchExpand(
            spatial_dims=spatial_dims,
            **patch_expansion_info[2],
            scale=scales[-3],
            norm_layer=nn.LayerNorm,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        self.concat_back_dim3d = self.get_concat_model(**concat_back_info[1])  # 128 56 56
        self.stage3d = LightMUNet(spatial_dims=spatial_dims, **decoder_layer_info[2], n_layers=5,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                                  add_last=True)  # 128 56 56
        self.patch_expand2d = PatchExpand(
            spatial_dims=spatial_dims,
            **patch_expansion_info[3],
            scale=scales[-4],
            norm_layer=nn.LayerNorm,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        self.concat_back_dim2d = self.get_concat_model(**concat_back_info[2])  # 64, 112, 112
        self.stage2d = LightMUNet(spatial_dims=spatial_dims, **decoder_layer_info[3], n_layers=6,
                                  input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                                  add_last=True)  # 64, 112, 112
        self.patch_expand1d = PatchExpand(
            spatial_dims=spatial_dims,
            **patch_expansion_info[4],
            scale=scales[-5],
            norm_layer=nn.LayerNorm,
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        self.concat_back_dim1d = self.get_concat_model(**concat_back_info[3])  # 64, 112, 112
        self.stage1d = LightMUNet(spatial_dims=spatial_dims, **decoder_layer_info[4], n_layers=7,
                                  input_patch_size=input_patch_size, add_last=True)

        self.side1 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side2 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side3 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side4 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side5 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side6 = Convolution(spatial_dims, 64, out_ch, kernel_size=3, padding=1, conv_only=True)

        self.outconv = Convolution(spatial_dims, 6 * out_ch, out_ch, kernel_size=1, conv_only=True)

    def get_concat_model(self, in_features: int, out_features: int):
        if in_features == out_features:
            return nn.Identity()
        else:
            return nn.Linear(in_features, out_features)

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
        # hx6up = _upsample_like(hx6, hx5.shape[2:])  # 512, 14, 14

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


def get_light_mamba2net_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        use_pretrain: bool = True,
        small_model=False
):
    # dim = len(configuration_manager.conv_kernel_sizes[0])
    # assert dim == 2, "Only 2D supported at the moment"
    label_manager = plans_manager.get_label_manager(dataset_json)
    if not small_model:
        model = LightMamba2Net(
            spatial_dims=len(configuration_manager.patch_size),
            input_patch_size=configuration_manager.patch_size,
            in_ch=num_input_channels,
            out_ch=label_manager.num_segmentation_heads,
            deep_supervision=deep_supervision)
    else:
        model = LightMamba2NetP(
            spatial_dims=len(configuration_manager.patch_size),
            input_patch_size=configuration_manager.patch_size,
            in_ch=num_input_channels,
            out_ch=label_manager.num_segmentation_heads,
            deep_supervision=deep_supervision)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


if __name__ == '__main__':
    # model = PatchEmbed2D(in_chans=3, embed_dim=96)
    # output = model(torch.rand(size=(1, 3, 224, 224)))
    # print(output.shape)
    from torchinfo import summary

    patch_size = [
        320,
        192
    ]
    in_ch = 3
    out_ch = 2
    model = LightMamba2Net(len(patch_size), in_ch, out_ch, True, input_patch_size=patch_size).to("cuda:0")
    output = model(torch.rand(size=(1, in_ch, *patch_size)).to("cuda:0"))
    for x_tensor in output:
        print(x_tensor.shape)

    summary(model, input_size=[1, in_ch] + patch_size, depth=5)

    model = LightMamba2Net(len(patch_size), in_ch, 1, True, input_patch_size=patch_size).to("cuda:0")
    output = model(torch.rand(size=(1, 3, *patch_size)).to("cuda:0"))
    for x_tensor in output:
        print(x_tensor.shape)
    print(model.scales)
    #
    # summary(model, input_size=[1, 3] + [224, 224])
    # print(output.shape)
