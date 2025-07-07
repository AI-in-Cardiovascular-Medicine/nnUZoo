import itertools
import numpy as np
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from monai.networks.nets import swin_unetr
import math
from functools import partial
from typing import Union, List, Tuple, Callable
from monai.utils import UpsampleMode, InterpolateMode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from monai.networks.blocks import Convolution
from monai.networks.blocks.upsample import UpSample

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


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


class SSND(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            factorization_type: str,
            d_model: int,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            dilation=1
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.spatial_dims = spatial_dims
        self.factorization_type = factorization_type
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.convnd = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            conv_only=True,
            dilation=dilation
        ).to(device)
        self.act = nn.SiLU()
        if self.factorization_type == "cross-scan" and spatial_dims == 2:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            )
        elif self.factorization_type == "cross-scan" and spatial_dims == 3:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  # w+
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  # w-
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  # H+
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  # H-
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  # Z+
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  # Z-
            )

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj
        if self.factorization_type == "cross-scan" and spatial_dims == 2:
            self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
            )
            self.k = len(self.dt_projs)
        elif self.factorization_type == "cross-scan" and spatial_dims == 3:
            self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                             **factory_kwargs),
            )
            self.k = len(self.dt_projs)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.k, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, Z, H, W = shape(x, self.spatial_dims)
        L = H * W * (Z or 1)
        K = self.k
        if self.factorization_type == "cross-scan" and self.spatial_dims == 2:
            x_hwwh = torch.stack([x.view(B, -1, L),
                                  torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                                 dim=1).view(B, self.spatial_dims, -1, L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        elif self.factorization_type == "cross-scan" and self.spatial_dims == 3:
            x_zhw = x.view(B, -1, L)
            x_wzh = rearrange(x, "b c z h w -> b c w z h").contiguous().view(B, -1, L)
            x_hwz = rearrange(x, "b c z h w -> b c h w z").contiguous().view(B, -1, L)

            x_hwwh = torch.stack([x_zhw, x_wzh, x_hwz], dim=1).view(B, self.spatial_dims, -1, L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        else:
            raise Exception("Factorization and spatial_dims are not supported!")
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)
        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        if self.factorization_type == "cross-scan" and self.spatial_dims == 2:
            inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, - 1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        elif self.factorization_type == "cross-scan" and self.spatial_dims == 3:
            inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, - 1, L)

            # x_wzh = rearrange(x, "b c z h w -> b c w z h")
            # x_hwz = rearrange(x, "b c z h w -> b c h w z")

            y_wzh = rearrange(out_y[:, 1].view(B, -1, W, Z, H), "b c w z h -> b c z h w").contiguous().view(B, -1, L)
            inv_y_wzh = rearrange(inv_y[:, 1].view(B, -1, W, Z, H), "b c w z h -> b c z h w").contiguous().view(B, -1,
                                                                                                                L)

            y_hwz = rearrange(out_y[:, 1].view(B, -1, W, Z, H), "b c h w z -> b c z h w").contiguous().view(B, -1, L)
            inv_y_hwz = rearrange(inv_y[:, 1].view(B, -1, W, Z, H), "b c h w z -> b c z h w").contiguous().view(B, -1,
                                                                                                                L)
            y = out_y[:, 0] + inv_y[:, 0] + y_wzh + inv_y_wzh + y_hwz + inv_y_hwz
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, Z, H, W, -1)
        else:
            raise Exception()
        return y

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = permute(x, self.spatial_dims, reverse=True).contiguous()
        x = self.act(self.convnd(x))  # (b, d, h, w)
        y = self.forward_core(x)
        # assert y1.dtype == torch.float32

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


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
    def _patch_merge3d(x: torch.tensor, zs: int, hs: int, ws: int):
        outputs = [[0] if zs == 1 else [0, 1],
                   [0] if hs == 1 else [0, 1],
                   [0] if ws == 1 else [0, 1],
                   ]
        x_outputs = []
        outputs = itertools.product(*outputs)
        for comb in outputs:
            xi = x[:, comb[0]::zs, comb[1]::hs, comb[2]::ws, :]
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
        if self.spatial_dims == 2:
            x0 = x[:, 0::self.hs, 0::self.ws, :]  # B H/self.scale W/self.scale C
            x1 = x[:, 1::self.hs, 0::self.ws, :]  # B H/self.scale W/self.scale C
            x2 = x[:, 0::self.hs, 1::self.ws, :]  # B H/self.scale W/self.scale C
            x3 = x[:, 1::self.hs, 1::self.ws, :]  # B H/self.scale W/self.scale C
            x = torch.cat([t for t in [x0, x1, x2, x3] if np.prod(t.shape) != 0], -1)  # B H/2 W/2 4*C
        elif self.spatial_dims == 3:
            x = self._patch_merge3d(x, self.zs, self.hs, self.ws)

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


class VSSMDecoder(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            factorization_type: str,
            num_classes: int,
            deep_supervision,
            features_per_stage: Union[Tuple[int, ...], List[int]] = None,
            depths: Union[Tuple[int, ...], List[int]] = None,
            drop_path_rate: float = 0.2,
            d_state: int = 16,
            patch_size: int = 4,
            scales=None,
            dilations: int = None,
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param depths:
        :param n_conv_per_stage:
        :param deep_supervision:
        :param skip_first_expand:

        """
        super().__init__()
        self.spatial_dims = spatial_dims
        # self.skip_first_expand = skip_first_expand
        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        # self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        self.depths = depths
        input_features_skip = 0  # will be overwritten, simply to skip IDE warning!
        # we start with the bottleneck and work out way up
        stages = []
        expand_layers = []
        seg_layers = []
        concat_back_dim = []
        dilations = dilations or [1] * (n_stages_encoder - 1)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            if scales is not None:
                scale = scales[-s]
                if np.prod(scale) == 1:
                    expand_layers.append(None)
                else:
                    expand_layers.append(PatchExpand(
                        spatial_dims=spatial_dims,
                        dim=input_features_below,
                        scale=scale,
                        output_dim=input_features_below,
                        norm_layer=nn.LayerNorm,
                    ))
            else:
                expand_layers.append(None)
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(VSSLayer(
                spatial_dims=spatial_dims,
                factorization_type=factorization_type,
                dim=input_features_skip,
                depth=1,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
                d_state=math.ceil(2 * input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
                dilation=dilations[s - 1],
            ))
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(Convolution(spatial_dims, input_features_skip, num_classes,
                                          1, 1, padding=0, bias=True, conv_only=True))
            concat_back_dim.append(nn.Linear(2 * input_features_skip, input_features_skip))

        if patch_size != 1:
            expand_layers.append(PatchExpand(
                spatial_dims=spatial_dims,
                dim=encoder_output_channels[0],
                scale=patch_size,
                norm_layer=nn.LayerNorm,
            ))
        else:
            expand_layers.append(None)
        stages.append(nn.Identity())
        seg_layers.append(
            Convolution(spatial_dims, input_features_skip, num_classes, 1, 1,
                        padding=0, bias=True, conv_only=True))
        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            if self.expand_layers[s] is None:
                x = permute(lres_input, self.spatial_dims)  # .permute(0, 2, 3, 1)
            else:
                x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, permute(skips[-(s + 2)], self.spatial_dims)), -1)
                x = self.concat_back_dim[s](x)
            x = permute(self.stages[s](x), self.spatial_dims, reverse=True)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    Reference: http://arxiv.org/abs/2401.10166
    """

    def __init__(self, spatial_dims: int = 2, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size) if spatial_dims == 2 else (patch_size, patch_size, patch_size)
        self.spatial_dims = spatial_dims
        self.proj = Convolution(spatial_dims=spatial_dims,
                                in_channels=in_chans,
                                out_channels=embed_dim,
                                kernel_size=patch_size,
                                strides=patch_size, conv_only=True)
        # nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        x = permute(x, self.spatial_dims)
        if self.norm is not None:
            x = self.norm(x)
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

        # self.proj4 = Convolution(spatial_dims, in_ch, in_ch, kernel_size=1, strides=1, padding=0,
        #                          conv_only=True)
        # self.norm4 = InstanceNorm(spatial_dims, in_ch)
        # self.nonliner4 = nn.ReLU()

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


class VSSBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            factorization_type: str,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            dilation: int = 1,
            **kwargs,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.ln_1 = norm_layer(hidden_dim)
        self.gsc = GSC(spatial_dims=spatial_dims, in_channels=hidden_dim)
        self.self_attention = SSND(spatial_dims=spatial_dims,
                                   factorization_type=factorization_type,
                                   d_model=hidden_dim,
                                   dropout=attn_drop_rate,
                                   d_state=d_state,
                                   dilation=dilation,
                                   **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input = permute(self.gsc(permute(input, self.spatial_dims, reverse=True)), self.spatial_dims)
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            spatial_dims: int,
            factorization_type: str,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            dilation: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                spatial_dims=spatial_dims,
                factorization_type=factorization_type,
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dilation=dilation,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSMEncoder(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 factorization_type: str,
                 patch_size=4,
                 in_chans=3,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 # skip_first_downsample: bool = False,
                 # skip_last_downsample: bool = False,
                 add_last: bool = False,
                 out_ch: int = None,
                 scales: List[Tuple[int, ...]] = None,
                 dilations: list = None,
                 ):
        super().__init__()
        self.scales = scales
        self.spatial_dims = spatial_dims
        self.num_layers = len(depths)
        self.add_last = add_last
        # self.skip_last_downsample = skip_last_downsample
        # self.skip_first_downsample = skip_first_downsample
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        if self.add_last:
            # self.rebnconvin = Convolution(spatial_dims, in_chans, out_ch, dilation=1)
            # self.rebnconvin = Convolution(spatial_dims, in_chans, out_ch, dilation=1,
            #                               strides=1, kernel_size=1, conv_only=True,
            #                               groups=gcd(in_chans, out_ch))
            self.rebnconvin = get_dwconv_layer(spatial_dims, in_chans, out_ch)
        self.embed_dim = dims[0]
        # self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(spatial_dims=spatial_dims,
                                        patch_size=patch_size,
                                        in_chans=out_ch if self.add_last else in_chans,
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dilations = dilations or [1] * self.num_layers
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                spatial_dims=spatial_dims,
                factorization_type=factorization_type,
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                # drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                downsample=None,
                use_checkpoint=use_checkpoint,
                dilation=dilations[i_layer]
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                # if i_layer == 0 and skip_first_downsample:
                #     continue
                #
                # if i_layer == (self.num_layers - 2) and skip_last_downsample:
                #     continue
                if scales is not None:
                    if np.prod(scales[i_layer]) != 1:
                        self.downsamples.append(PatchMerging2D(spatial_dims=spatial_dims,
                                                               input_dim=dims[i_layer],
                                                               scale=scales[i_layer],
                                                               output_features=dims[i_layer + 1],
                                                               norm_layer=norm_layer))
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = []
        if self.add_last:
            x = self.rebnconvin(x)
            x_ret.append(x)
        else:
            x_ret.append(None)

        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(permute(x, self.spatial_dims, reverse=True))
            if s < len(self.downsamples):
                # if s == 0 and self.skip_first_downsample:
                #     continue
                x = self.downsamples[s](x)

        return x_ret


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


class MU(nn.Module):

    def __init__(self,
                 spatial_dims: int,
                 factorization_type: str,
                 in_ch: int, mid_ch,
                 out_ch: int,
                 n_layers: int,
                 patch_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 4,
                 add_last: bool = False,
                 input_patch_size: Union[Tuple[int, ...]] = None,
                 vss_args: dict = None,
                 decoder_args: dict = None,
                 ):
        super().__init__()
        self.add_last = add_last
        self.input_patch_size = input_patch_size
        features = [mid_ch] * n_layers
        depths = [2] * n_layers

        scales = get_scales(spatial_dims, input_patch_size, n_layers - 1, patch_size)
        self.scales = scales
        vss_args = dict(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_chans=in_ch,
            patch_size=patch_size,
            depths=depths,
            dims=features,
            add_last=add_last,
            out_ch=out_ch if add_last else None,
            scales=scales,
            drop_path_rate=0.2,
            **(vss_args or dict())
        )

        decoder_args = dict(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            num_classes=out_ch,
            deep_supervision=False,
            features_per_stage=features,
            drop_path_rate=0.2,
            d_state=16,
            depths=depths,
            scales=scales,
            patch_size=patch_size,
            **(decoder_args or dict())
        )

        self.vssm_encoder = VSSMEncoder(**vss_args)
        self.vssm_decoder = VSSMDecoder(**decoder_args)

    def forward(self, x):
        skips = self.vssm_encoder(x)
        out = self.vssm_decoder(skips)
        if self.add_last:
            out = out + skips[0]
        return out

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
class SSND2Net(nn.Module):

    def __init__(self,
                 spatial_dims: int,
                 factorization_type: str,
                 in_ch: int, out_ch: int, deep_supervision: bool,
                 input_patch_size):
        super(SSND2Net, self).__init__()
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.input_patch_size = input_patch_size
        scales = get_scales(spatial_dims, input_patch_size, n_layers=5, patch_size=None)
        self.scales = scales
        self.stage1 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type,
                         in_ch=in_ch,
                         mid_ch=16,
                         out_ch=32,
                         n_layers=7,
                         input_patch_size=input_patch_size,
                         patch_size=1,
                         add_last=True)
        # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.patch_merging1 = PatchMerging2D(spatial_dims, 32, scale=scales[0],
                                             output_features=64)  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type, in_ch=64, mid_ch=32, out_ch=64, n_layers=6,
                         input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                         patch_size=1,
                         add_last=True)
        self.patch_merging2 = PatchMerging2D(spatial_dims, 64, scale=scales[1],
                                             output_features=128)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type, in_ch=128, mid_ch=64, out_ch=128, n_layers=5,
                         patch_size=1,
                         input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                         add_last=True)
        self.patch_merging3 = PatchMerging2D(spatial_dims, 128, scale=scales[2],
                                             output_features=256)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type, in_ch=256, mid_ch=128, out_ch=256, n_layers=4,
                         patch_size=1,
                         input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                         add_last=True)
        self.patch_merging4 = PatchMerging2D(spatial_dims, 256, scale=scales[3],
                                             output_features=512)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = MU(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_ch=512, mid_ch=256, out_ch=512, n_layers=4,
            patch_size=1,
            add_last=True,
            # vss_args={"dilations": [1, 2, 4, 8]},
            # decoder_args={"dilations": [4, 2, 1]}
        )
        self.patch_merging5 = PatchMerging2D(spatial_dims, 512,
                                             scale=scales[4],
                                             output_features=512)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage6 = MU(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_ch=512, mid_ch=256, out_ch=512, n_layers=4,
            patch_size=1,
            add_last=True,
            # vss_args={"dilations": [1, 2, 4, 8]},
            # decoder_args={"dilations": [4, 2, 1]}
        )
        # spatial_dims, 512, 256, 512)  # in: 512, 7, 7 -> 512, 7, 7

        # decoder
        self.patch_expand5d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=512,
            scale=scales[-1],
            norm_layer=nn.LayerNorm,
            output_dim=512
        )  # 512, 14, 14 -> 512, 28, 28
        # -> concat -> 512, 28, 28
        self.stage5d = MU(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_ch=1024, mid_ch=256, out_ch=512, n_layers=4,
            patch_size=1,
            add_last=True,
            # vss_args={"dilations": [1, 2, 4, 8]},
            # decoder_args={"dilations": [4, 2, 1]}
        )
        # spatial_dims, 1024, 256, 512)  # in: 1024, 14, 14 -> 512, 14, 14

        self.patch_expand4d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=512,
            scale=scales[-2],
            norm_layer=nn.LayerNorm,
            output_dim=256
        )
        # 512, 14, 14 -> 256, 28, 28
        # -> concat -> 512, 28, 28

        self.concat_back_dim4d = nn.Linear(512, 256)

        self.stage4d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=256, mid_ch=128, out_ch=256, n_layers=4,
                          patch_size=1,
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
        self.stage3d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=128, mid_ch=64, out_ch=128, n_layers=5,
                          input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                          patch_size=1,
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
        self.stage2d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=64, mid_ch=32, out_ch=64, n_layers=6,
                          input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                          patch_size=1,
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
        self.stage1d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=32, mid_ch=16, out_ch=32, n_layers=7,
                          input_patch_size=input_patch_size,
                          patch_size=1,
                          add_last=True)

        self.side1 = Convolution(spatial_dims, 32, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side2 = Convolution(spatial_dims, 64, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side3 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side4 = Convolution(spatial_dims, 256, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side5 = Convolution(spatial_dims, 512, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side6 = Convolution(spatial_dims, 512, out_ch, kernel_size=3, padding=1, conv_only=True)

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
        self.spatial_up_sample = PatchExpand(
            spatial_dims=spatial_dims,
            dim=in_ch,
            scale=2,
            norm_layer=nn.LayerNorm,
            output_dim=in_ch,
        )
        self.feature_down_sample_1 = get_dwconv_layer(spatial_dims=spatial_dims, in_channels=in_ch,
                                                      out_channels=in_ch // 2)
        # self.feature_down_sample_2 = get_dwconv_layer(spatial_dims=spatial_dims, in_ch=in_ch * 2,
        #                                               out_ch=in_ch)

    def forward(self, enc_x, x):
        x = self.spatial_up_sample(x).permute(0, 3, 1, 2)
        x = self.feature_down_sample_1(x)
        x = torch.cat([enc_x, x], dim=1)
        # x = self.feature_down_sample_2(x)
        return x


class SSND2NetP(nn.Module):

    def __init__(self,
                 spatial_dims: int,
                 factorization_type: str,
                 in_ch: int, out_ch: int, deep_supervision: bool,
                 input_patch_size):
        super(SSND2NetP, self).__init__()
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.input_patch_size = input_patch_size
        scales = get_scales(spatial_dims, input_patch_size, n_layers=5, patch_size=None)
        self.scales = scales
        self.stage1 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type,
                         in_ch=in_ch,
                         mid_ch=16,
                         out_ch=64,
                         n_layers=7,
                         input_patch_size=input_patch_size,
                         patch_size=1,
                         add_last=True)
        # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.patch_merging1 = PatchMerging2D(spatial_dims, 64, scale=scales[0],
                                             output_features=64)  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type, in_ch=64, mid_ch=16, out_ch=64, n_layers=6,
                         input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                         patch_size=1,
                         add_last=True)
        self.patch_merging2 = PatchMerging2D(spatial_dims, 64, scale=scales[1],
                                             output_features=64)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type, in_ch=64, mid_ch=16, out_ch=64, n_layers=5,
                         patch_size=1,
                         input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                         add_last=True)
        self.patch_merging3 = PatchMerging2D(spatial_dims, 64, scale=scales[2],
                                             output_features=64)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = MU(spatial_dims=spatial_dims,
                         factorization_type=factorization_type, in_ch=64, mid_ch=16, out_ch=64, n_layers=4,
                         patch_size=1,
                         input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                         add_last=True)
        self.patch_merging4 = PatchMerging2D(spatial_dims, 64, scale=scales[3],
                                             output_features=64)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = MU(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_ch=64, mid_ch=16, out_ch=64, n_layers=4,
            patch_size=1,
            add_last=True,
            # vss_args={"dilations": [1, 2, 4, 8]},
            # decoder_args={"dilations": [4, 2, 1]}
        )
        self.patch_merging5 = PatchMerging2D(spatial_dims, 64,
                                             scale=scales[4],
                                             output_features=64)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage6 = MU(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_ch=64, mid_ch=16, out_ch=64, n_layers=4,
            patch_size=1,
            add_last=True,
            # vss_args={"dilations": [1, 2, 4, 8]},
            # decoder_args={"dilations": [4, 2, 1]}
        )
        # spatial_dims, 512, 256, 512)  # in: 512, 7, 7 -> 512, 7, 7

        # decoder
        self.patch_expand5d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=64,
            scale=scales[-1],
            norm_layer=nn.LayerNorm,
            output_dim=64
        )  # 512, 14, 14 -> 512, 28, 28
        # -> concat -> 512, 28, 28
        self.stage5d = MU(
            spatial_dims=spatial_dims,
            factorization_type=factorization_type,
            in_ch=128, mid_ch=16, out_ch=128, n_layers=4,
            patch_size=1,
            add_last=True,
            # vss_args={"dilations": [1, 2, 4, 8]},
            # decoder_args={"dilations": [4, 2, 1]}
        )
        # spatial_dims, 1024, 256, 512)  # in: 1024, 14, 14 -> 512, 14, 14

        self.patch_expand4d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=128,
            scale=scales[-2],
            norm_layer=nn.LayerNorm,
            output_dim=64
        )  # 512, 14, 14 -> 256, 28, 28
        # -> concat -> 512, 28, 28

        self.concat_back_dim4d = nn.Linear(128, 128)

        self.stage4d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=128, mid_ch=16, out_ch=128, n_layers=4,
                          patch_size=1,
                          input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:3]),
                          add_last=True)
        self.patch_expand3d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=128,
            scale=scales[-3],
            norm_layer=nn.LayerNorm,
            output_dim=64,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        self.concat_back_dim3d = nn.Linear(128, 128)  # 128 56 56
        self.stage3d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=128, mid_ch=16, out_ch=128, n_layers=5,
                          input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:2]),
                          patch_size=1,
                          add_last=True)  # 128 56 56
        self.patch_expand2d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=128,
            scale=scales[-4],
            norm_layer=nn.LayerNorm,
            output_dim=64,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        self.concat_back_dim2d = nn.Linear(128, 128)  # 64, 112, 112
        self.stage2d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=128, mid_ch=16, out_ch=128, n_layers=6,
                          input_patch_size=get_scale_value(spatial_dims, input_patch_size, scales[:1]),
                          patch_size=1,
                          add_last=True)  # 64, 112, 112
        self.patch_expand1d = PatchExpand(
            spatial_dims=spatial_dims,
            dim=128,
            scale=scales[-5],
            norm_layer=nn.LayerNorm,
            output_dim=64
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        self.concat_back_dim1d = nn.Linear(128, 128)  # 64, 112, 112
        self.stage1d = MU(spatial_dims=spatial_dims,
                          factorization_type=factorization_type, in_ch=128, mid_ch=16, out_ch=128, n_layers=7,
                          input_patch_size=input_patch_size,
                          patch_size=1,
                          add_last=True)

        self.side1 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side2 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side3 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side4 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side5 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side6 = Convolution(spatial_dims, 64, out_ch, kernel_size=3, padding=1, conv_only=True)

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


def get_m2net_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        use_pretrain: bool = True
):
    # dim = len(configuration_manager.conv_kernel_sizes[0])
    # assert dim == 2, "Only 2D supported at the moment"
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = SSND2Net(
        spatial_dims=len(configuration_manager.patch_size),
        factorization_type="cross-scan",
        in_ch=num_input_channels,
        out_ch=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        input_patch_size=configuration_manager.patch_size
    )
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


def get_ssnd2net_from_plans(
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
        model = SSND2Net(
            spatial_dims=len(configuration_manager.patch_size),
            factorization_type="cross-scan",
            in_ch=num_input_channels,
            out_ch=label_manager.num_segmentation_heads,
            deep_supervision=deep_supervision,
            input_patch_size=configuration_manager.patch_size
        )
    else:
        model = SSND2NetP(
            spatial_dims=len(configuration_manager.patch_size),
            factorization_type="cross-scan",
            in_ch=num_input_channels,
            out_ch=label_manager.num_segmentation_heads,
            deep_supervision=deep_supervision,
            input_patch_size=configuration_manager.patch_size
        )
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    return model


if __name__ == '__main__':
    from torchinfo import summary

    in_ch = 1
    out_ch = 1
    input_patch_size_ = [256, 256]
    model = SSND2NetP(spatial_dims=len(input_patch_size_), factorization_type="cross-scan", in_ch=in_ch, out_ch=out_ch,
                     deep_supervision=True,
                     input_patch_size=input_patch_size_).to("cuda:0")
    output = model(torch.rand(size=(1, in_ch, *input_patch_size_)).to("cuda:0"))
    for x in output:
        print(x.shape)
    summary(model, input_size=[1, in_ch] + input_patch_size_)
