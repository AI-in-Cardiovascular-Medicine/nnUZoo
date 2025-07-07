from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from einops import rearrange
from torchinfo import summary
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from nnunetv2.nets.mask_funcs import window_masking, patchify, unpatchify

from nnunetv2.utilities.network_initialization import InitWeights_He


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
                             strides=1, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


class MambaLayer(nn.Module):
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


def get_mamba_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


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
        :param order:
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


class ResUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
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


class LightMUNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 32,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
            mae: bool = False,
            mask_ratio: float = 0.0
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.mae = mae


        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        if self.mae:
            self.mask_ratio = mask_ratio
            self.mask_token = nn.Parameter(torch.zeros(1, 1, init_filters))
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

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
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, order=orders[i % len(orders)])
                  for _ in range(item)]
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
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
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

        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward_mae_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = patchify(imgs, patch_size, in_chans=in_ch)
        # print(f"target shape: {target.shape}")
        # if norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5
        # mask = rearrange(mask, "")
        loss = (pred - imgs) ** 2
        # print("loss shape: ", loss.shape)
        loss = loss.mean(dim=1).view((loss.shape[0], -1))  # get the mean over all the features(48) of each patch
        # print("loss shape: ", loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # Only the ones that are masked are considered not the original ones!
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        imgs = x
        x = self.convInit(x)
        if self.mae:
            x, mask = window_masking(x.to(torch.float32),
                                     self.mask_token,
                                     remove=False,
                                     mask_len_sparse=False,
                                     mask_ratio=self.mask_ratio,
                                     input_shape = "B C H W")
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        if self.mae:
            loss = self.forward_mae_loss(imgs, x, mask )
            return loss, x, mask
        else:
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


def get_from_plans(
        spatial_dims: int,
        in_ch: int,
        out_ch: int,
        small_mode=False,
        mae: bool = False,
        mask_ratio: float = 0,
        **kwargs
):
    # dim = len(configuration_manager.conv_kernel_sizes[0])
    # assert dim == 2, "Only 2D supported at the moment"
    if not small_mode:
        model = LightMUNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            mae=mae,
            mask_ratio=mask_ratio
        )

    else:
        raise NotImplementedError()
        # model = MambaND2NetP(
        #     spatial_dims=len(configuration_manager.patch_size),
        #     factorization_type="cross-scan",
        #     in_ch=num_input_channels,
        #     out_ch=label_manager.num_segmentation_heads,
        #     deep_supervision=deep_supervision,
        #     input_patch_size=configuration_manager.patch_size
        # )

    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


if __name__ == '__main__':
    in_channels = 1
    bs = 2
    patch_size = [320, 320]
    out_channels = 5
    model = LightMUNet(spatial_dims=len(patch_size), init_filters=8, in_channels=in_channels,
                       out_channels=out_channels)
    summary(model, input_size=[1, in_channels, *patch_size],
            # depth=5
            )
    inputs = torch.rand((bs, in_channels, *patch_size)).to("cuda:0")
    output = model(inputs)
    print(output.shape)

