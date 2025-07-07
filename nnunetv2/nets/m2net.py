import math
from functools import partial
from typing import Union, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from monai.networks.blocks import Convolution
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from timm.models.layers import DropPath, trunc_normal_


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


def _upsample_like(src, tar_shape):
    src = F.upsample(src, size=tar_shape, mode='bilinear')

    return src


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
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
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

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
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

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
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

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

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
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


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
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
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
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
    def __init__(self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 skip_first_downsample: bool = False,
                 skip_last_downsample: bool = False,
                 add_last: bool = False,
                 out_ch: int = None
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.add_last = add_last
        self.skip_last_downsample = skip_last_downsample
        self.skip_first_downsample = skip_first_downsample
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        if self.add_last:
            self.rebnconvin = REBNCONV(in_chans, out_ch, dirate=1)
        self.embed_dim = dims[0]
        # self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size,
                                        in_chans=out_ch if self.add_last else in_chans,
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
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
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                if i_layer == 0 and skip_first_downsample:
                    continue

                if i_layer == (self.num_layers - 2) and skip_last_downsample:
                    continue
                self.downsamples.append(PatchMerging2D(input_dim=dims[i_layer],
                                                       scale=2,
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
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                if s == 0 and self.skip_first_downsample:
                    continue
                x = self.downsamples[s](x)

        return x_ret


class MU(nn.Module):

    def __init__(self, in_ch: int, mid_ch, out_ch: int, n_layers: int, skip_last_downsample: bool = False,
                 patch_size: int = 4,
                 add_last: bool = False):
        super().__init__()
        self.add_last = add_last
        features = [mid_ch] * n_layers
        depths = [1] * n_layers
        vss_args = dict(
            in_chans=in_ch,
            patch_size=patch_size,
            depths=depths,
            dims=features,
            skip_first_downsample=False,
            skip_last_downsample=skip_last_downsample,
            add_last=add_last,
            out_ch=out_ch if add_last else None,
            # dims=96,
            drop_path_rate=0.2
        )

        decoder_args = dict(
            num_classes=out_ch,
            deep_supervision=False,
            features_per_stage=features,
            drop_path_rate=0.2,
            d_state=16,
            depths=depths,
            skip_first_expand=skip_last_downsample,
            patch_size=patch_size
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
class M2Net(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, deep_supervision: bool):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.stage1 = MU(in_ch=in_ch, mid_ch=16, out_ch=32, n_layers=7, skip_last_downsample=True,
                         patch_size=1, add_last=True)
        # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.patch_merging1 = PatchMerging2D(32, scale=2)  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = MU(in_ch=64, mid_ch=32, out_ch=64, n_layers=6, patch_size=1,
                         skip_last_downsample=True, add_last=True)
        self.patch_merging2 = PatchMerging2D(64, scale=2)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = MU(in_ch=128, mid_ch=64, out_ch=128, n_layers=5, patch_size=1,
                         skip_last_downsample=True, add_last=True)
        self.patch_merging3 = PatchMerging2D(128, scale=2)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = MU(in_ch=256, mid_ch=128, out_ch=256, n_layers=4, patch_size=1,
                         skip_last_downsample=True, add_last=True)
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

        self.stage4d = MU(in_ch=256, mid_ch=128, out_ch=256, n_layers=4, patch_size=1,
                          skip_last_downsample=True, add_last=True)
        self.patch_expand3d = PatchExpand(
            dim=256,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        self.concat_back_dim3d = nn.Linear(256, 128)  # 128 56 56
        self.stage3d = MU(in_ch=128, mid_ch=64, out_ch=128, n_layers=5, patch_size=1,
                          skip_last_downsample=True, add_last=True)  # 128 56 56
        self.patch_expand2d = PatchExpand(
            dim=128,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        self.concat_back_dim2d = nn.Linear(128, 64)  # 64, 112, 112
        self.stage2d = MU(in_ch=64, mid_ch=32, out_ch=64, n_layers=6, patch_size=1,
                          skip_last_downsample=True, add_last=True)  # 64, 112, 112
        self.patch_expand1d = PatchExpand(
            dim=64,
            scale=2,
            norm_layer=nn.LayerNorm,
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        self.concat_back_dim1d = nn.Linear(64, 32)  # 64, 112, 112
        self.stage1d = MU(in_ch=32, mid_ch=16, out_ch=32, n_layers=7, patch_size=1,
                          skip_last_downsample=True, add_last=True)

        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

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


class M2NetP(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, deep_supervision: bool, spatial_dims: int = 2):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.stage1 = MU(in_ch=in_ch, mid_ch=16, out_ch=64, n_layers=7, skip_last_downsample=True,
                         patch_size=1, add_last=True)
        # self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.patch_merging1 = PatchMerging2D(64, scale=2, output_features=64)  # in: 32, 224, 224 -> out: 64, 112, 112

        self.stage2 = MU(in_ch=64, mid_ch=16, out_ch=64, n_layers=6, patch_size=1,
                         skip_last_downsample=True, add_last=True)
        self.patch_merging2 = PatchMerging2D(64, scale=2, output_features=64)  # in: 64, 112, 112 -> out: 128, 56, 56

        self.stage3 = MU(in_ch=64, mid_ch=16, out_ch=64, n_layers=5, patch_size=1,
                         skip_last_downsample=True, add_last=True)
        self.patch_merging3 = PatchMerging2D(64, scale=2, output_features=64)  # in: 128, 56, 56 -> out: 256, 28, 28

        self.stage4 = MU(in_ch=64, mid_ch=16, out_ch=64, n_layers=4, patch_size=1,
                         skip_last_downsample=True, add_last=True)
        self.patch_merging4 = PatchMerging2D(64, scale=2, output_features=64)  # in: 256, 28, 28 -> out: 512, 14, 14

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # in: 512, 14, 14 -> out: 512, 7, 7

        self.stage6 = RSU4F(64, 16, 64)  # in: 512, 7, 7 -> 512, 7, 7

        # decoder
        self.stage5d = RSU4F(128, 16, 128)  # in: 1024, 14, 14 -> 512, 14, 14
        # self.patch_expand4d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
        self.patch_expand4d = PatchExpand(
            dim=128,
            scale=2,
            norm_layer=nn.LayerNorm,
            # output_dim=64,
        )  # 128, 14, 14 -> 64, 28, 28
        # -> concat -> 128, 28, 28

        # self.concat_back_dim4d = nn.Linear(512, 256)

        self.stage4d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=4, patch_size=1,
                          skip_last_downsample=True, add_last=True)
        # self.patch_expand3d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
        self.patch_expand3d = PatchExpand(
            dim=128,
            scale=2,
            norm_layer=nn.LayerNorm,
            # output_dim=64,
        )  # 256, 28, 28 -> 128, 56, 56
        # concat -> 256, 56, 56
        # self.concat_back_dim3d = nn.Linear(256, 128)  # 128 56 56
        self.stage3d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=5, patch_size=1,
                          skip_last_downsample=True, add_last=True)  # 128 56 56
        # self.patch_expand2d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
        self.patch_expand2d = PatchExpand(
            dim=128,
            scale=2,
            norm_layer=nn.LayerNorm,
            # output_dim=128,
        )  # 128, 56, 56 -> 64, 112, 112
        # concat -> 128, 112, 112
        # self.concat_back_dim2d = nn.Linear(128, 64)  # 64, 112, 112
        self.stage2d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=6, patch_size=1,
                          skip_last_downsample=True, add_last=True)  # 64, 112, 112
        # self.patch_expand1d = UnetDecoderUpSampleLayer(spatial_dims=spatial_dims, in_ch=128)
        self.patch_expand1d = PatchExpand(
            dim=128,
            scale=2,
            norm_layer=nn.LayerNorm,
            # output_dim=128,
        )  # 64, 112, 112 -> 32, 224, 224
        # concat -> 64, 224, 224
        # self.concat_back_dim1d = nn.Linear(64, 32)  # 64, 112, 112
        self.stage1d = MU(in_ch=128, mid_ch=16, out_ch=128, n_layers=7, patch_size=1,
                          skip_last_downsample=True, add_last=True)

        self.side1 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

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
        hx5dup = torch.cat([hx5dup.permute(0, 3, 1, 2), hx4], 1)
        # 512, 14, 14 -> 256, 28, 28

        hx4d = self.stage4d(hx5dup)
        hx4dup = self.patch_expand3d(hx4d)
        hx4dup = torch.cat([hx4dup.permute(0, 3, 1, 2), hx3], 1)
        # 512, 14, 14 -> 256, 28, 28

        hx3d = self.stage3d(hx4dup)
        hx3dup = self.patch_expand2d(hx3d)
        hx3dup = torch.cat([hx3dup.permute(0, 3, 1, 2), hx2], 1)
        # 512, 14, 14 -> 256, 28, 28
        hx2d = self.stage2d(hx3dup)
        hx2dup = self.patch_expand1d(hx2d)
        hx2dup = torch.cat([hx2dup.permute(0, 3, 1, 2), hx1], 1)
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


def get_m2net_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True,
        use_pretrain: bool = True
):
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = M2Net(in_ch=num_input_channels,
                  out_ch=label_manager.num_segmentation_heads,
                  deep_supervision=deep_supervision)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)


    return model


def get_m2netp_from_plans(
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

    model = M2NetP(in_ch=num_input_channels,
                   out_ch=label_manager.num_segmentation_heads,
                   deep_supervision=deep_supervision)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)


    return model


if __name__ == '__main__':
    from torchinfo import summary

    patch_size = [
        320,
        192
    ]
    model = M2Net(3, 2, True).to("cuda:0")
    output = model(torch.rand(size=(1, 3, *patch_size)).to("cuda:0"))
    for x in output:
        print(x.shape)

    summary(model, input_size=[1, 3] + patch_size)
