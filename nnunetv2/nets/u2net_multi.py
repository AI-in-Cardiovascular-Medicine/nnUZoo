from monai.utils import UpsampleMode, InterpolateMode

import torch
import torch.nn as nn
from monai.networks.blocks.upsample import UpSample

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from monai.networks.blocks.convolutions import Convolution
from nnunetv2.nets.mask_funcs import gen_random_mask, upsample_mask, patchify, unpatchify, unpatchify_mask


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


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
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


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = Convolution(spatial_dims, in_ch, out_ch, dilation=1)

        self.rebnconv1 = Convolution(spatial_dims, out_ch, mid_ch, dilation=1)
        self.pool1 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv2 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1)
        self.pool2 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv3 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1)
        self.pool3 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv4 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1)
        self.pool4 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv5 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1)
        self.pool5 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv6 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1)

        self.rebnconv7 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=2)

        self.rebnconv6d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1)
        self.rebnconv5d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1)
        self.rebnconv4d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1)
        self.rebnconv3d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1)
        self.rebnconv2d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1)
        self.rebnconv1d = Convolution(spatial_dims, mid_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        # print(hx1d.shape)
        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, mid_ch=12, out_ch=3, act='relu', norm='BATCH'):
        super(RSU6, self).__init__()

        self.rebnconvin = Convolution(spatial_dims, in_ch, out_ch, dilation=1, act=act, norm=norm)

        self.rebnconv1 = Convolution(spatial_dims, out_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool1 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv2 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool2 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv3 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool3 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv4 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool4 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv5 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)

        self.rebnconv6 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=2, act=act, norm=norm)

        self.rebnconv5d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv4d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv3d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv2d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv1d = Convolution(spatial_dims, mid_ch * 2, out_ch, dilation=1, act=act, norm=norm)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, mid_ch=12, out_ch=3, act='relu', norm='BATCH'):
        super(RSU5, self).__init__()

        self.rebnconvin = Convolution(spatial_dims, in_ch, out_ch, dilation=1, act=act, norm=norm)

        self.rebnconv1 = Convolution(spatial_dims, out_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool1 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv2 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool2 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv3 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool3 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv4 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)

        self.rebnconv5 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=2, act=act, norm=norm)

        self.rebnconv4d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv3d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv2d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv1d = Convolution(spatial_dims, mid_ch * 2, out_ch, dilation=1, act=act, norm=norm)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, mid_ch=12, out_ch=3, act: str = "relu", norm: str = "BATCH"):
        super(RSU4, self).__init__()

        self.rebnconvin = Convolution(spatial_dims, in_ch, out_ch, dilation=1, act=act, norm=norm)

        self.rebnconv1 = Convolution(spatial_dims, out_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool1 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv2 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.pool2 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.rebnconv3 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=1, act=act, norm=norm)

        self.rebnconv4 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=2, act=act, norm=norm)

        self.rebnconv3d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv2d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv1d = Convolution(spatial_dims, mid_ch * 2, out_ch, dilation=1, act=act, norm=norm)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, mid_ch=12, out_ch=3, act="relu", norm="BATCH"):
        super(RSU4F, self).__init__()

        self.rebnconvin = Convolution(spatial_dims, in_ch, out_ch, dilation=1, act=act, norm=norm)

        self.rebnconv1 = Convolution(spatial_dims, out_ch, mid_ch, dilation=1, act=act, norm=norm)
        self.rebnconv2 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=2, act=act, norm=norm)
        self.rebnconv3 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=4, act=act, norm=norm)

        self.rebnconv4 = Convolution(spatial_dims, mid_ch, mid_ch, dilation=8, act=act, norm=norm)

        self.rebnconv3d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=4, act=act, norm=norm)
        self.rebnconv2d = Convolution(spatial_dims, mid_ch * 2, mid_ch, dilation=2, act=act, norm=norm)
        self.rebnconv1d = Convolution(spatial_dims, mid_ch * 2, out_ch, dilation=1, act=act, norm=norm)

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


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, out_ch=1, deep_supervision=False):
        super(U2NET, self).__init__()
        self.deep_supervision = deep_supervision
        self.unpatchify = unpatchify
        self.stage1 = RSU7(spatial_dims, in_ch, 32, 64)
        self.pool12 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(spatial_dims, 64, 32, 128)
        self.pool23 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(spatial_dims, 128, 64, 256)
        self.pool34 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(spatial_dims, 256, 128, 512)
        self.pool45 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(spatial_dims, 512, 256, 512)
        self.pool56 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(spatial_dims, 512, 256, 512)

        # decoder
        self.stage5d = RSU4F(spatial_dims, 1024, 256, 512)
        self.stage4d = RSU4(spatial_dims, 1024, 128, 256)
        self.stage3d = RSU5(spatial_dims, 512, 64, 128)
        self.stage2d = RSU6(spatial_dims, 256, 32, 64)
        self.stage1d = RSU7(spatial_dims, 128, 16, 64)

        self.side1 = Convolution(spatial_dims, 64, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side2 = Convolution(spatial_dims, 64, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side3 = Convolution(spatial_dims, 128, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side4 = Convolution(spatial_dims, 256, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side5 = Convolution(spatial_dims, 512, out_ch, kernel_size=3, padding=1, conv_only=True)
        self.side6 = Convolution(spatial_dims, 512, out_ch, kernel_size=3, padding=1, conv_only=True)

        self.outconv = Convolution(spatial_dims, 6 * out_ch, out_ch, kernel_size=1, conv_only=True)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)  # 1, 64, 256, 256
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)  # 1, 128, 128, 128

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)  # 1, 512, 8, 8
        hx6up = _upsample_like(hx6, hx5)  #

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # 1, 512, 16, 16
        hx5dup = _upsample_like(hx5d, hx4)  # 1, 512, 32, 32

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))  # 1, 64, 256, 256

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        if self.deep_supervision:
            return d0, d1, d2, d3, d4, d5, d6
        else:
            return d0

    @torch.no_grad()
    def freeze_encoder(self):
        for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.pool12,
                      self.pool23, self.pool34, self.pool45, self.pool56]:
            for name, param in group.named_parameters():
                # if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.pool12,
                      self.pool23, self.pool34, self.pool45, self.pool56]:
            for param in group.parameters():
                param.requires_grad = True


### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self, spatial_dims: int = 2, in_ch=3, out_ch=1, deep_supervision=False, mae=False,
                 mask_ratio=0.75,
                 # init_filters: int = 32
                 ):
        super(U2NETP, self).__init__()
        self.in_ch = in_ch
        self.deep_supervision = deep_supervision
        self.mae = mae

        if self.mae:
            self.unpatchify_mask = unpatchify_mask
            self.patchify_size = 16
            self.mask_ratio = mask_ratio
            self.mask_token = nn.Parameter(torch.zeros(1, 128, 1, 1))
            # self.init_filters = init_filters
            #
            # self.stem = nn.Sequential(
            #     nn.Conv2d(in_ch, init_filters, kernel_size=4, stride=4),
            #     nn.LayerNorm(init_filters, eps=1e-6)
            # )
            # in_ch = init_filters

        self.stage1 = RSU7(spatial_dims, in_ch, 16, 64)
        self.pool12 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(spatial_dims, 64, 16, 64)
        self.pool23 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(spatial_dims, 64, 16, 64)
        self.pool34 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(spatial_dims, 64, 16, 64)
        self.pool45 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(spatial_dims, 64, 16, 64)
        self.pool56 = MaxPool(spatial_dims, 2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(spatial_dims, 64, 16, 64)

        # decoder
        output_features = [128 if self.mae else 64 for _ in range(5)]
        self.stage5d = RSU4F(spatial_dims, 128, 16, output_features[0])
        self.stage4d = RSU4(spatial_dims, 128, 16, output_features[1])
        self.stage3d = RSU5(spatial_dims, 128, 16, output_features[2])
        self.stage2d = RSU6(spatial_dims, 128, 16, output_features[3])
        self.stage1d = RSU7(spatial_dims, 128, 16, output_features[4])

        self.side1 = Convolution(spatial_dims, output_features[0], out_ch, kernel_size=3, padding=1)
        self.side2 = Convolution(spatial_dims, output_features[1], out_ch, kernel_size=3, padding=1)
        self.side3 = Convolution(spatial_dims, output_features[2], out_ch, kernel_size=3, padding=1)
        self.side4 = Convolution(spatial_dims, output_features[3], out_ch, kernel_size=3, padding=1)
        self.side5 = Convolution(spatial_dims, output_features[4], out_ch, kernel_size=3, padding=1)
        self.side6 = Convolution(spatial_dims, output_features[4], out_ch, kernel_size=3, padding=1)

        self.outconv = Convolution(spatial_dims, 6 * out_ch, out_ch, kernel_size=1, conv_only=True)

    def forward_mae_loss(self, imgs, pred, mask, norm_pix_loss):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """

        # print("pred:", pred.shape)
        # if len(pred.shape) == 4:
        #     n, c, _, _ = pred.shape
        #     pred = pred.reshape(n, -1, c)
        # pred = torch.einsum('ncl->nlc', pred) #.reshape(pred.shape[0], -1)
        pred = patchify(pred, self.patchify_size, self.in_ch)
        target = patchify(imgs, self.patchify_size, self.in_ch)
        # print("mask:", mask.shape,"pred:", pred.shape,"target:", target.shape)
        # target = target.reshape(target.shape[0], -1)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print("loss:", loss.shape, "mask:", mask.shape, "pred:", pred.shape, "target:", target.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        imgs = x
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        if self.mae:
            mask_gen = gen_random_mask(x, self.mask_ratio, self.patchify_size)
            mask = upsample_mask(mask_gen, 2 ** (3 - 1))  # 4 stages are left!
            mask = mask.unsqueeze(1).type_as(x)
            hx *= (1. - mask)
        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        staged5_input = torch.cat((hx6up, hx5), 1)
        if self.mae:
            n, c, h, w = staged5_input.shape
            mask = mask_gen.reshape(-1, h, w).unsqueeze(1).type_as(x)
            mask_token = self.mask_token.repeat(staged5_input.shape[0], 1, staged5_input.shape[2],
                                                staged5_input.shape[3])
            staged5_input = staged5_input * (1. - mask) + mask_token * mask

        # decoder
        hx5d = self.stage5d(staged5_input)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(hx5dup if self.mae else torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(hx4dup if self.mae else torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(hx3dup if self.mae else torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(hx2dup if self.mae else torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(staged5_input if self.mae else hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        if self.deep_supervision:
            return d0, d1, d2, d3, d4, d5, d6
        else:
            if self.mae:
                loss = self.forward_mae_loss(imgs, d0, mask_gen, False)
                return loss, d0, mask_gen
            else:
                return d0
            # return

    @torch.no_grad()
    def freeze_encoder(self):
        for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.pool12,
                      self.pool23, self.pool34, self.pool45, self.pool56]:
            for name, param in group.named_parameters():
                # if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for group in [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6, self.pool12,
                      self.pool23, self.pool34, self.pool45, self.pool56]:
            for param in group.parameters():
                param.requires_grad = True


def get_u2netp_from_plans(
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

    model = U2NETP(spatial_dims=len(configuration_manager.patch_size),
                   in_ch=num_input_channels, out_ch=label_manager.num_segmentation_heads,
                   deep_supervision=deep_supervision)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


def get_from_plans(
        spatial_dims,
        in_ch: int,
        out_ch,
        deep_supervision: bool = True,
        use_pretrain: bool = True,
        mae: bool = True,
        mask_ratio: bool = 0,
):
    model = U2NETP(spatial_dims=spatial_dims,
                   in_ch=in_ch, out_ch=out_ch,
                   deep_supervision=deep_supervision,
                   mae=mae,
                   mask_ratio=mask_ratio)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


# num_segmentation_heads: int,
# num_input_channels: int,
# deep_supervision: bool = True,
# use_pretrain: bool = True
def get_u2net_from_plans(
        spatial_dims: int,
        num_segmentation_heads: int,
        num_input_channels: int,
        deep_supervision: bool = True,
        use_pretrain: bool = True
):
    # dim = len(configuration_manager.conv_kernel_sizes[0])
    # assert dim == 2, "Only 2D supported at the moment"

    model = U2NET(spatial_dims=spatial_dims,
                  in_ch=num_input_channels, out_ch=num_segmentation_heads,
                  deep_supervision=deep_supervision)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    # if use_pretrain:
    #     model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model


if __name__ == '__main__':
    input_data = torch.rand(size=(1, 3, 256, 256)).to("cpu")
    model = U2NETP(spatial_dims=len(input_data.shape[2:]), in_ch=3, out_ch=5, mae=True).to("cpu")
    output = model(input_data)
    print(output.shape)
