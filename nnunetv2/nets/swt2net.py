from timm.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import math
from functools import partial
from typing import Union, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.layers import DropPath, trunc_normal_
from monai.networks.blocks import Convolution


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = get_dwconv_layer(spatial_dims=2, in_channels=in_ch, out_channels=out_ch)
        # self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


def _upsample_like(src, tar_shape):
    src = F.upsample(src, size=tar_shape, mode='bilinear')

    return src


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer. The output will have scale times features and H and W will be divided by scale times.
    Args:
        input_dim (int): Resolution of input feature.
        scale (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_dim: int, scale: int, output_features: int = None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_feature_size = (scale ** 2) * input_dim
        self.output_features = output_features or input_dim * scale
        self.scale = scale
        self.reduction = nn.Linear(self.input_feature_size, self.output_features, bias=False)
        self.norm = norm_layer(self.input_feature_size)

    def forward(self, x, permute=False):
        if permute:
            x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % self.scale != 0) or (H % self.scale != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // self.scale
            SHAPE_FIX[1] = W // self.scale

        x0 = x[:, 0::self.scale, 0::self.scale, :]  # B H/self.scale W/self.scale C
        x1 = x[:, 1::self.scale, 0::self.scale, :]  # B H/self.scale W/self.scale C
        x2 = x[:, 0::self.scale, 1::self.scale, :]  # B H/self.scale W/self.scale C
        x3 = x[:, 1::self.scale, 1::self.scale, :]  # B H/self.scale W/self.scale C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // self.scale, W // self.scale, (self.scale ** 2) * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        if permute:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, dim: int, scale, output_dim: int = None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.output_dim = output_dim
        if self.output_dim is None:
            self.expand = nn.Linear(dim, self.scale * dim, bias=False)
            self.norm = norm_layer(dim // self.scale)
        else:
            self.expand = nn.Linear(dim // (self.scale ** 2), self.output_dim, bias=False)
            self.norm = norm_layer(self.output_dim)

    def forward(self, x, permute=False):
        if self.output_dim is None:
            x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
            x = self.expand(x)
            B, H, W, C = x.shape

            # x = x.view(B, H, W, C)
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.scale, p2=self.scale,
                          c=C // (self.scale ** 2))

            x = x.view(B, -1, C // (self.scale ** 2))
            x = self.norm(x)
            x = x.reshape(B, H * self.scale, W * self.scale, C // (self.scale ** 2))
        else:
            x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
            B, H, W, C = x.shape

            # x = x.view(B, H, W, C)
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.scale, p2=self.scale,
                          c=C // (self.scale ** 2))
            x = self.expand(x)  # new shape!
            x = x.view(B, -1, self.output_dim)
            x = self.norm(x)
            x = x.reshape(B, H * self.scale, W * self.scale, self.output_dim)
        if permute:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x  # .permute(0, 3, 1, 2)


class VSSMDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            deep_supervision,
            features_per_stage: Union[Tuple[int, ...], List[int]] = None,
            depths: Union[Tuple[int, ...], List[int]] = None,
            drop_path_rate: float = 0.2,
            d_state: int = 16,
            skip_first_expand: bool = False,
            patch_size: int = 4,
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
        self.skip_first_expand = skip_first_expand
        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        # self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = depths or [2] * len(features_per_stage)
        input_features_skip = 0  # will be overwritten, simply to skip IDE warning!
        # we start with the bottleneck and work out way up
        stages = []
        expand_layers = []
        seg_layers = []
        concat_back_dim = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            if s == 1 and self.skip_first_expand:
                expand_layers.append(None)
            else:
                expand_layers.append(PatchExpand(
                    dim=input_features_below,
                    scale=2,
                    output_dim=input_features_below,
                    norm_layer=nn.LayerNorm,
                ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=1,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
                d_state=math.ceil(2 * input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            ))
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(nn.Linear(2 * input_features_skip, input_features_skip))

        # for final prediction
        # expand_layers.append(FinalPatchExpand_X4(
        #     dim=encoder_output_channels[0],
        #     dim_scale=4,
        #     norm_layer=nn.LayerNorm,
        # ))
        expand_layers.append(PatchExpand(
            dim=encoder_output_channels[0],
            scale=patch_size,
            norm_layer=nn.LayerNorm,
        ))
        stages.append(nn.Identity())
        seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))

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
            if s == 0 and self.skip_first_expand:
                x = lres_input.permute(0, 2, 3, 1)
            else:
                x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
            x = self.stages[s](x).permute(0, 3, 1, 2)
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

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from functools import partial

from nnunetv2.utilities.network_initialization import InitWeights_He
from typing import Optional
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as func
from einops import rearrange
from monai.networks.blocks import Convolution


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


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    @staticmethod
    def _pad(left_x, right_x):
        if left_x.shape != right_x:
            pass

    def forward(self, x):
        _, H, W, _ = x.shape
        pad = H % self.window_size != 0 or W % self.window_size != 0
        if pad:
            x = F.pad(x, (0, 0, self.window_size - W % self.window_size, 0, self.window_size - H % self.window_size, 0))
        x_copy = x
        x = self.norm1(x)

        x = self.attn(x)
        x = self.drop_path(x)
        x = x + x_copy

        x_copy = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x_copy
        if pad:
            # Drop the first ones! Because padding puts at the left part of the input image.
            x = x[:, -H:, -W:, :]
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 4, in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size,) * 2, stride=(patch_size,) * 2)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = func.pad(x, (0, self.patch_size - W % self.patch_size,
                             0, self.patch_size - H % self.patch_size,
                             0, 0))
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm, patch_size: int = 4):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, (patch_size ** 2) * dim, bias=False)
        self.norm = norm_layer(dim)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=self.patch_size, P2=self.patch_size)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: Optional[bool] = True,
                 attn_drop: Optional[float] = 0., proj_drop: Optional[float] = 0., shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_size = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_size, coords_size], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        x = rearrange(x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C', Nh=H // self.window_size, Nw=W // self.window_size)
        return x

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

        img_mask = torch.zeros((1, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, H, W, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self.create_mask(x)
        else:
            mask = None

        x = self.window_partition(x)
        Bn, Mh, Mw, _ = x.shape
        x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')
        qkv = rearrange(self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C', Nh=H // Mh, Nw=W // Mw)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    @staticmethod
    def _pad(left_x, right_x):
        if left_x.shape != right_x:
            pass

    def forward(self, x):
        _, H, W, _ = x.shape
        pad = H % self.window_size != 0 or W % self.window_size != 0
        if pad:
            x = F.pad(x, (0, 0, self.window_size - W % self.window_size, 0, self.window_size - H % self.window_size, 0))
        x_copy = x
        x = self.norm1(x)

        x = self.attn(x)
        x = self.drop_path(x)
        x = x + x_copy

        x_copy = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x_copy
        if pad:
            # Drop the first ones! Because padding puts at the left part of the input image.
            x = x[:, -H:, -W:, :]
        return x


class BasicBlock(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging: bool = True):
        super(BasicBlock, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        if patch_merging:
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicBlockUp(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm):
        super(BasicBlockUp, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.upsample(x)
        return x


class SwinTransformerUnet(nn.Module):
    def __init__(self, patch_size: int = 4, in_ch: int = 3, out_ch: int = 1000, embed_dim: int = 96,
                 window_size: int = 7, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True, add_last: bool = False):
        super().__init__()
        self.add_last = add_last
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer
        if self.add_last:
            self.rebnconvin = get_dwconv_layer(2, in_ch, out_ch)

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, in_c=in_ch, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = self.build_layers()
        self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.skip_connection_layers = self.skip_connection()
        self.norm_up = norm_layer(embed_dim)
        self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer, patch_size=patch_size)
        self.head = nn.Conv2d(in_channels=embed_dim, out_channels=out_ch, kernel_size=(1, 1), bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up

    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers

    @staticmethod
    def pad(l_x, r_x):
        # if r_x.shape < l_x.shape:
        # r_x = F.pad(r_x, (0, 0, l_x.shape[2] - r_x.shape[2], 0, l_x.shape[1] - r_x.shape[1], 0))
        # else:
        # This will remove the extra samples from the input x
        l_x = l_x[:, :r_x.shape[1], :r_x.shape[2], :]
        return l_x, r_x

    def forward(self, x):
        if self.add_last:
            last_add = self.rebnconvin(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

        x = self.first_patch_expanding(x)

        for i, layer in enumerate(self.layers_up):
            prev_x_save = x_save[len(x_save) - i - 2]
            x, prev_x_save = self.pad(x, prev_x_save)
            x = torch.cat([x, prev_x_save], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        x = self.norm_up(x)
        x = self.final_patch_expanding(x)

        x = rearrange(x, 'B H W C -> B C H W')
        x = self.head(x)
        if self.add_last:
            x = x + last_add
        return x


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### M^2-Net ####
class SwT2Net(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, deep_supervision: bool):
        super().__init__()
        spatial_dims = 2
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.stage1 = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=4,
            in_ch=in_ch,
            out_ch=32,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=32, num_heads=(2, 2, 4, 8),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.patch_merging1 = PatchMerging2D(32, scale=2)  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=4,
            in_ch=64,
            out_ch=64,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=64, num_heads=(2, 4, 8, 16),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        self.patch_merging2 = PatchMerging2D(64, scale=2)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=2,
            in_ch=128,
            out_ch=128,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        self.patch_merging3 = PatchMerging2D(128, scale=2)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=1,
            in_ch=256,
            out_ch=256,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        self.patch_merging4 = PatchMerging2D(256, scale=2)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # in: 512, 14, 14 -> out: 512, 7, 7

        self.stage6 = RSU4F(512, 256, 512)  # in: 512, 7, 7 -> 512, 7, 7

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)  # in: 1024, 14, 14 -> 512, 14, 14

        self.patch_expand4d = PatchExpand(
            dim=512,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 512, 14, 14 -> 256, 28, 28
        # -> concat -> 512, 28, 28

        self.concat_back_dim4d = nn.Linear(512, 256)

        self.stage4d = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=1,
            in_ch=256,
            out_ch=256,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        self.patch_expand3d = PatchExpand(
            dim=256,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        self.concat_back_dim3d = nn.Linear(256, 128)  # 128 56 56
        self.stage3d = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=2,
            in_ch=128,
            out_ch=128,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        self.patch_expand2d = PatchExpand(
            dim=128,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        self.concat_back_dim2d = nn.Linear(128, 64)  # 64, 112, 112
        self.stage2d = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=4,
            in_ch=64,
            out_ch=64,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=64, num_heads=(2, 4, 8, 16),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
        self.patch_expand1d = PatchExpand(
            dim=64,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        self.concat_back_dim1d = nn.Linear(64, 32)  # 64, 112, 112
        self.stage1d = SwinTransformerUnet(
            # img_size=configuration_manager.patch_size,
            patch_size=4,
            in_ch=32,
            out_ch=32,
            # decoder_embed_dim=768,
            depths=(2, 2, 4, 2), embed_dim=32, num_heads=(2, 2, 4, 8),
            window_size=7, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            add_last=True
        )
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
        hx = self.patch_merging1(hx1, permute=True)  # 64, 112, 112

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.patch_merging2(hx2, permute=True)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.patch_merging3(hx3, permute=True)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.patch_merging4(hx4, permute=True)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5.shape[2:])  # 512, 14, 14

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # in: 1024, 14, 14 -> out: 512, 14, 14
        hx5dup = self.patch_expand4d(hx5d)  # 512, 14, 14 -> 256, 28, 28
        hx5dup = self.concat_back_dim4d(torch.cat((hx5dup, hx4.permute(0, 2, 3, 1)), -1)).permute(0, 3, 1, 2)
        # 512, 14, 14 -> 256, 28, 28

        hx4d = self.stage4d(hx5dup)
        hx4dup = self.patch_expand3d(hx4d)
        hx4dup = self.concat_back_dim3d(torch.cat((hx4dup, hx3.permute(0, 2, 3, 1)), -1)).permute(0, 3, 1, 2)
        # 512, 14, 14 -> 256, 28, 28

        hx3d = self.stage3d(hx4dup)
        hx3dup = self.patch_expand2d(hx3d)
        hx3dup = self.concat_back_dim2d(torch.cat((hx3dup, hx2.permute(0, 2, 3, 1)), -1)).permute(0, 3, 1, 2)
        # 512, 14, 14 -> 256, 28, 28
        hx2d = self.stage2d(hx3dup)
        hx2dup = self.patch_expand1d(hx2d)
        hx2dup = self.concat_back_dim1d(torch.cat((hx2dup, hx1.permute(0, 2, 3, 1)), -1)).permute(0, 3, 1, 2)
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

        d0 = self.outconv(torch.cat((d1, _upsample_like(d2, d1.shape[2:]), _upsample_like(d3, d1.shape[2:]),
                                     _upsample_like(d4, d1.shape[2:]), _upsample_like(d5, d1.shape[2:]),
                                     _upsample_like(d6, d1.shape[2:])), 1))

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


# class M2NetP(nn.Module):
#
#     def __init__(self, in_ch: int, out_ch: int, deep_supervision: bool, spatial_dims: int = 2):
#         super().__init__()
#         self.deep_supervision = deep_supervision
#         self.stage1 = MU(in_ch=in_ch, mid_ch=16, out_ch=64, n_layers=7, skip_last_downsample=True,
#                          patch_size=1, add_last=True)
#         # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#         self.patch_merging1 = PatchMerging2D(64, scale=2, output_features=64)  # in: 32, 224, 224 -> out: 64, 112, 112
#
#         self.stage2 = MU(in_ch=64, mid_ch=16, out_ch=64, n_layers=6, patch_size=1,
#                          skip_last_downsample=True, add_last=True)
#         self.patch_merging2 = PatchMerging2D(64, scale=2, output_features=64)  # in: 64, 112, 112 -> out: 128, 56, 56
#
#         self.stage3 = MU(in_ch=64, mid_ch=16, out_ch=64, n_layers=5, patch_size=1,
#                          skip_last_downsample=True, add_last=True)
#         self.patch_merging3 = PatchMerging2D(64, scale=2, output_features=64)  # in: 128, 56, 56 -> out: 256, 28, 28
#
#         self.stage4 = MU(in_ch=64, mid_ch=16, out_ch=64, n_layers=4, patch_size=1,
#                          skip_last_downsample=True, add_last=True)
#         self.patch_merging4 = PatchMerging2D(64, scale=2, output_features=64)  # in: 256, 28, 28 -> out: 512, 14, 14
#
#         self.stage5 = RSU4F(64, 16, 64)
#         self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # in: 512, 14, 14 -> out: 512, 7, 7
#
#         self.stage6 = RSU4F(64, 16, 64)  # in: 512, 7, 7 -> 512, 7, 7
#
#         # decoder
#         self.stage5d = RSU4F(128, 16, 128)  # in: 1024, 14, 14 -> 512, 14, 14
#         # self.patch_expand4d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
#         self.patch_expand4d = PatchExpand(
#             dim=128,
#             scale=2,
#             norm_layer=nn.LayerNorm,
#             # output_dim=64,
#         )  # 128, 14, 14 -> 64, 28, 28
#         # -> concat -> 128, 28, 28
#
#         # self.concat_back_dim4d = nn.Linear(512, 256)
#
#         self.stage4d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=4, patch_size=1,
#                           skip_last_downsample=True, add_last=True)
#         # self.patch_expand3d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
#         self.patch_expand3d = PatchExpand(
#             dim=128,
#             scale=2,
#             norm_layer=nn.LayerNorm,
#             # output_dim=64,
#         )  # 256, 28, 28 -> 128, 56, 56
#         # concat -> 256, 56, 56
#         # self.concat_back_dim3d = nn.Linear(256, 128)  # 128 56 56
#         self.stage3d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=5, patch_size=1,
#                           skip_last_downsample=True, add_last=True)  # 128 56 56
#         # self.patch_expand2d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
#         self.patch_expand2d = PatchExpand(
#             dim=128,
#             scale=2,
#             norm_layer=nn.LayerNorm,
#             # output_dim=128,
#         )  # 128, 56, 56 -> 64, 112, 112
#         # concat -> 128, 112, 112
#         # self.concat_back_dim2d = nn.Linear(128, 64)  # 64, 112, 112
#         self.stage2d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=6, patch_size=1,
#                           skip_last_downsample=True, add_last=True)  # 64, 112, 112
#         # self.patch_expand1d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
#         self.patch_expand1d = PatchExpand(
#             dim=128,
#             scale=2,
#             norm_layer=nn.LayerNorm,
#             # output_dim=128,
#         )  # 64, 112, 112 -> 32, 224, 224
#         # concat -> 64, 224, 224
#         # self.concat_back_dim1d = nn.Linear(64, 32)  # 64, 112, 112
#         self.stage1d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=7, patch_size=1,
#                           skip_last_downsample=True, add_last=True)
#
#         self.side1 = nn.Conv2d(128, out_ch, 3, padding=1)
#         self.side2 = nn.Conv2d(128, out_ch, 3, padding=1)
#         self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
#         self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)
#         self.side5 = nn.Conv2d(128, out_ch, 3, padding=1)
#         self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
#
#         self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
#
#     def forward(self, x):
#         hx = x
#
#         # stage 1
#         hx1 = self.stage1(hx)  # 32, 224, 224
#         hx = self.patch_merging1(hx1, permute=True)  # 64, 112, 112
#
#         # stage 2
#         hx2 = self.stage2(hx)
#         hx = self.patch_merging2(hx2, permute=True)
#
#         # stage 3
#         hx3 = self.stage3(hx)
#         hx = self.patch_merging3(hx3, permute=True)
#
#         # stage 4
#         hx4 = self.stage4(hx)
#         hx = self.patch_merging4(hx4, permute=True)
#
#         # stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)
#
#         # stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample_like(hx6, hx5.shape[2:])  # 512, 14, 14
#
#         # -------------------- decoder --------------------
#         hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # in: 1024, 14, 14 -> out: 512, 14, 14
#         hx5dup = self.patch_expand4d(hx5d)  # 512, 14, 14 -> 256, 28, 28
#         hx5dup = torch.cat([hx5dup.permute(0, 3, 1, 2), hx4], 1)
#         # 512, 14, 14 -> 256, 28, 28
#
#         hx4d = self.stage4d(hx5dup)
#         hx4dup = self.patch_expand3d(hx4d)
#         hx4dup = torch.cat([hx4dup.permute(0, 3, 1, 2), hx3], 1)
#         # 512, 14, 14 -> 256, 28, 28
#
#         hx3d = self.stage3d(hx4dup)
#         hx3dup = self.patch_expand2d(hx3d)
#         hx3dup = torch.cat([hx3dup.permute(0, 3, 1, 2), hx2], 1)
#         # 512, 14, 14 -> 256, 28, 28
#         hx2d = self.stage2d(hx3dup)
#         hx2dup = self.patch_expand1d(hx2d)
#         hx2dup = torch.cat([hx2dup.permute(0, 3, 1, 2), hx1], 1)
#         hx1d = self.stage1d(hx2dup)
#
#         # side output
#         d1 = self.side1(hx1d)
#
#         d2 = self.side2(hx2d)
#         # d2 = _upsample_like(d2, d1.shape[2:])
#
#         d3 = self.side3(hx3d)
#         # d3 = _upsample_like(d3, d1.shape[2:])
#
#         d4 = self.side4(hx4d)
#         # d4 = _upsample_like(d4, d1.shape[2:])
#
#         d5 = self.side5(hx5d)
#         # d5 = _upsample_like(d5, d1.shape[2:])
#
#         d6 = self.side6(hx6)
#         # d6 = _upsample_like(d6, d1.shape[2:])
#
#         d0 = self.outconv(torch.cat((d1, _upsample_like(d2, d1.shape[2:]), _upsample_like(d3, d1.shape[2:]),
#                                      _upsample_like(d4, d1.shape[2:]), _upsample_like(d5, d1.shape[2:]),
#                                      _upsample_like(d6, d1.shape[2:])), 1))
#
#         # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
#         if self.deep_supervision:
#             return d0, d1, d2, d3, d4, d5, d6
#         else:
#             return d0
#
#     @torch.no_grad()
#     def freeze_encoder(self):
#         for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.patch_merging1,
#                       self.patch_merging2, self.patch_merging3, self.patch_merging4, self.pool56]:
#             for name, param in group.named_parameters():
#                 # if "patch_embed" not in name:
#                 param.requires_grad = False
#
#     @torch.no_grad()
#     def unfreeze_encoder(self):
#         for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.patch_merging1,
#                       self.patch_merging2, self.patch_merging3, self.patch_merging4, self.pool56]:
#             for param in group.parameters():
#                 param.requires_grad = True


def get_swt2net_from_plans(
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

    model = SwT2Net(in_ch=num_input_channels,
                    out_ch=label_manager.num_segmentation_heads,
                    deep_supervision=deep_supervision)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


# def get_m2netp_from_plans(
#         plans_manager: PlansManager,
#         dataset_json: dict,
#         configuration_manager: ConfigurationManager,
#         num_input_channels: int,
#         deep_supervision: bool = True,
#         use_pretrain: bool = True
# ):
#     # dim = len(configuration_manager.conv_kernel_sizes[0])
#     # assert dim == 2, "Only 2D supported at the moment"
#     label_manager = plans_manager.get_label_manager(dataset_json)
#
#     model = M2NetP(in_ch=num_input_channels,
#                    out_ch=label_manager.num_segmentation_heads,
#                    deep_supervision=deep_supervision)
#     model.apply(InitWeights_He(1e-2))
#     model.apply(init_last_bn_before_add_to_0)
#
#     # if use_pretrain:
#     #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)
#
#     return model


if __name__ == '__main__':
    # model = PatchEmbed2D(in_chans=3, embed_dim=96)
    # output = model(torch.rand(size=(1, 3, 224, 224)))
    # print(output.shape)
    from torchinfo import summary

    patch_size = [
        256,
        224
    ]
    model = SwT2Net(3, 2, True).to("cuda:0")
    output = model(torch.rand(size=(1, 3, *patch_size)).to("cuda:0"))
    for x in output:
        print(x.shape)

    summary(model, input_size=[1, 3] + patch_size)
    # print(output.shape)
