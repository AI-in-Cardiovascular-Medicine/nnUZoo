from typing import Literal
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from monai.networks.nets import SwinUNETR
from nnunetv2.nets.segmentation.u2net_multi import get_u2net_from_plans
from ctaug import RandomArtifactTransform, StepMotionTransform
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary

try:
    from nnunetv2.nets.segmentation.SwinUMamba import get_swin_umamba_from_plans
except Exception as e:
    pass


class nnUNetTrainerMetalArtifactStepMotion(nnUNetTrainer):

    def __init__(
            self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda'), num_epochs: int = 1000, initial_lr=0.01,
            batch_size: int = 6, **kwargs,
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs,
                         initial_lr=initial_lr,
                         batch_size=batch_size, **kwargs)

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Before choosing the patch-size it's better not to have any augmentation since they may not be part of the chosen patch :)
        # transforms.append(RandomTransform(
        #     MetalArtifactGenerationStepMotion(spacing=self.configuration_manager.spacing, exclude_labels=(0, 1, 10, 9)
        #                                  ), apply_probability=0.1
        # ))

        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        transforms.append(RandomTransform(
            StepMotionTransform(
                motion_move_range=(0.01, 0.025),
            ), apply_probability=0.1
        ))
        # Make sure metal artifact is after -1 to prevent choosing -1 as the label :(
        transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                    exclude_labels=(0, 1, 10, 9),
                                    verbose=True,
                                    ), apply_probability=0.15
        ))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    def get_validation_transforms(
            self,
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        transforms.append(RandomTransform(
            StepMotionTransform(
                motion_move_range=(0.01, 0.025),
            ), apply_probability=0.1
        ))

        transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                    exclude_labels=(0, 1, 10, 9),
                                    verbose=True,
                                    ), apply_probability=0.1
        ))

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)


class nnUNetTrainerMetalArtifactStepMotionNoMirror(nnUNetTrainerMetalArtifactStepMotion):

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerMetalArtifactStepMotionNoMirror_1000(nnUNetTrainerMetalArtifactStepMotionNoMirror):
    """
    Extra test, remove it later
    """
    pass


class nnUNetTrainerMetalArtifactStepMotionNoMirrorU2NetS(nnUNetTrainerMetalArtifactStepMotionNoMirror):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         **kwargs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.spatial_dims: int = len(self.configuration_manager.patch_size)
        # self.spatial_dims: int = 3

    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            configuration_manager: ConfigurationManager,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
            add_skip_connection: bool = True,
            ssl_finetune_with_skip_connection: bool = False,
            up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
            **kwargs,
    ) -> nn.Module:

        model = get_u2net_from_plans(
            num_output_channels,
            num_input_channels,
            spatial_dims=len(configuration_manager.patch_size),
            deep_supervision=enable_deep_supervision,
            small=True
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            if self.spatial_dims == 2:
                deep_supervision_scales = [[1.0, 1.0]] * 7
            elif self.spatial_dims == 3:
                deep_supervision_scales = [[1.0, 1.0, 1.0]] * 7
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled


class nnUNetTrainerMetalArtifactStepMotionNoMirrorU2Net(nnUNetTrainerMetalArtifactStepMotionNoMirrorU2NetS):

    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            configuration_manager: ConfigurationManager,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
            add_skip_connection: bool = True,
            ssl_finetune_with_skip_connection: bool = False,
            up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
            **kwargs,
    ) -> nn.Module:
        model = get_u2net_from_plans(
            num_output_channels,
            num_input_channels,
            spatial_dims=len(configuration_manager.patch_size),
            deep_supervision=enable_deep_supervision,
            small=False
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model


class nnUNetTrainerMetalArtifactStepMotionNoMirrorSwUNETR(nnUNetTrainerMetalArtifactStepMotionNoMirror):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         **kwargs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.spatial_dims: int = len(self.configuration_manager.patch_size)
        # self.spatial_dims: int = 3

    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            configuration_manager: ConfigurationManager,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
            add_skip_connection: bool = True,
            ssl_finetune_with_skip_connection: bool = False,
            up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
            **kwargs,
    ) -> nn.Module:
        model = SwinUNETR(
            # img_size=configuration_manager.patch_size[0],
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=len(configuration_manager.patch_size),
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0, 1.0]] * 7
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled

    def plot_network_architecture(self):
        print('No Plotting')


class nnUNetTrainerMetalArtifactStepMotionNoMirrorSwinUMamba(nnUNetTrainerMetalArtifactStepMotionNoMirror):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         **kwargs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.spatial_dims: int = len(self.configuration_manager.patch_size)

        # self.spatial_dims: int = 3

    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            configuration_manager: ConfigurationManager,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
            add_skip_connection: bool = True,
            ssl_finetune_with_skip_connection: bool = False,
            up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
            **kwargs,
    ) -> nn.Module:
        model = get_swin_umamba_from_plans(
            num_input_channels,
            num_output_channels,
            
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
            spatial_dims=3,
            feat_size=[48, 96, 192, 384, 768]
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            if self.spatial_dims == 2:
                deep_supervision_scales = [[1.0, 1.0], [0.5, 0.5], [0.25, 0.25], [0.125, 0.125]]
            else:
                deep_supervision_scales = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled

    def plot_network_architecture(self):
        print('No Plotting')

