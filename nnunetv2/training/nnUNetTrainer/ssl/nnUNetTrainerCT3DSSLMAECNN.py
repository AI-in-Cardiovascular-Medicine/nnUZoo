import os
from os.path import join
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.utilities.file_and_folder_operations import join
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from ctaug.metal.transforms import RandomArtifactTransform
from ctaug.step.transforms import StepMotionTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.data_augmentation.ssl_augmentations.mae_augmentation import FullPreprocessing
from nnunetv2.training.nnUNetTrainer.ssl.nnUNetTrainerSSL import nnUNetTrainerSSL
from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization, ClippedZScoreNormalization
from nnunetv2.training.dataloading.nnunet_dataset import MonaiDatasetNumpyNiiGz


class nnUNetTrainerCT3DSSLMAENNUNetP256_BS96(nnUNetTrainerSSL):
    def __init__(
            self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda'), num_epochs: int = 1000, initial_lr=0.01,
            batch_size: int = 96, **kwargs,
    ):
        batch_size = batch_size or 96
        plans["configurations"][configuration]["patch_size"] = (256, 256, 256)
        plans["configurations"][configuration]["batch_size"] = batch_size
        plans["configurations"][configuration]["architecture"][
            "network_class_name"] = "nnunetv2.nets.dynamic_network_architectures.unet.ResidualEncoderUNet"
        plans["configurations"][configuration]["architecture"]["arch_kwargs"] = {"n_stages": 6,
                                                                                 "features_per_stage": [32,
                                                                                                        64,
                                                                                                        128,
                                                                                                        256,
                                                                                                        320,
                                                                                                        320],
                                                                                 "conv_op": "torch.nn.modules.conv.Conv3d",
                                                                                 "kernel_sizes": [
                                                                                     [3, 3, 3],
                                                                                     [3, 3, 3],
                                                                                     [3, 3, 3],
                                                                                     [3, 3, 3],
                                                                                     [3, 3, 3],
                                                                                     [3, 3, 3]],
                                                                                 "strides": [[1, 1, 1],
                                                                                             [2, 2, 2],
                                                                                             [2, 2, 2],
                                                                                             [2, 2, 2],
                                                                                             [2, 2, 2],
                                                                                             [2, 2, 2]],
                                                                                 "n_blocks_per_stage": [1,
                                                                                                        3,
                                                                                                        4,
                                                                                                        6,
                                                                                                        6,
                                                                                                        6],
                                                                                 "n_conv_per_stage_decoder": [
                                                                                     1, 1, 1, 1, 1
                                                                                 ],
                                                                                 "conv_bias": True,
                                                                                 "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                                                                                 "norm_op_kwargs": {
                                                                                     "eps": 1e-05,
                                                                                     "affine": True},
                                                                                 "dropout_op": None,
                                                                                 "dropout_op_kwargs": None,
                                                                                 "nonlin": "torch.nn.LeakyReLU",
                                                                                 "nonlin_kwargs": {
                                                                                     "inplace": True}
                                                                                 }
        super().__init__(plans,
                         configuration,
                         fold,
                         dataset_json,
                         unpack_dataset,
                         device,
                         num_epochs,
                         initial_lr=initial_lr,
                         batch_size=batch_size,
                         **kwargs)
        # self.spatial_size = 3
        self.mask_ratio = 0.75
        self.enable_deep_supervision = False
        self.im_output_folder = join(self.output_folder, "img_log")
        os.makedirs(self.im_output_folder, exist_ok=True)
        self.apply_qualitative_per_n_epochs = 50
        self.apply_quantitative_per_n_epochs = 50
        self.do_best_qualitative = True
        self.do_best_quantitative = True
        self.add_skip_connection = False
        self.mask_block_size: int = 16


class nnUNetTrainerCT3DSSLMAENNUNetArtifactAugP256_BS96(nnUNetTrainerCT3DSSLMAENNUNetP256_BS96):

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> AbstractTransform:
        val_transforms = []
        # val_transforms.append(NumpyToTensor(["data"], "float"))
        val_transforms.append(RandomTransform(
            StepMotionTransform(), apply_probability=0.05
        ))

        val_transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                         # exclude_labels=(0, 1, 10, 9),
                                         skip_minus_one_selected_label=False,
                                         verbose=False,
                                         ), apply_probability=0.05
        ))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            # ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            # ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0,
                random_crop=False, p_elastic_deform=0,
                p_rotation=0.1,
                rotation=rotation_for_DA, p_scaling=0.1, scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        # if mirror_axes is not None and len(mirror_axes) > 0:
        #     tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(RandomTransform(
            StepMotionTransform(),
            apply_probability=0.05
        ))

        tr_transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                         # exclude_labels=(0, 1, 10, 9),
                                         skip_minus_one_selected_label=False,
                                         verbose=False,
                                         ), apply_probability=0.05
        ))
        # tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms


class nnUNetTrainerCT3DSSLMAENNUNetArtifactAugP256_BS2(nnUNetTrainerCT3DSSLMAENNUNetArtifactAugP256_BS96):
    def __init__(
            self, **kwargs,
    ):
        kwargs["batch_size"] = 2
        super().__init__(**kwargs)


class nnUNetTrainerCT3DSSLMAENNUNetMonaiDLP256_BS96(nnUNetTrainerCT3DSSLMAENNUNetP256_BS96):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_preprocessor=FullPreprocessing(
                    dataset_json=self.dataset_json,
                    plans_manager=self.plans_manager,
                    configuration_manager=self.configuration_manager,
                    normalization_cls=CTNormalization
                    )
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        
        dataset_tr = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=tr_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        dataset_val = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        
        return dataset_tr, dataset_val
    
    
    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> AbstractTransform:
        val_transforms = []
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            # ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0,
                random_crop=False, p_elastic_deform=0,
                p_rotation=0.1,
                rotation=rotation_for_DA, p_scaling=0.1, scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms = Compose(tr_transforms)
        return tr_transforms
    
    
class nnUNetTrainerCT3DSSLMAENNUNetArtifactAugMonaiDLP256_ClippedZScore_BS2(nnUNetTrainerCT3DSSLMAENNUNetP256_BS96):
    def __init__(self, **kwargs):
        kwargs["batch_size"] = 2
        super().__init__(**kwargs)
        self.data_preprocessor=FullPreprocessing(
            dataset_json=self.dataset_json,
                    plans_manager=self.plans_manager,
                    configuration_manager=self.configuration_manager,
                    normalization_cls=ClippedZScoreNormalization
                    )
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        
        dataset_tr = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=tr_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        dataset_val = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        
        return dataset_tr, dataset_val
    
        
    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> AbstractTransform:
        val_transforms = []
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            # ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0,
                random_crop=False, p_elastic_deform=0,
                p_rotation=0.1,
                rotation=rotation_for_DA, p_scaling=0.1, scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(RandomTransform(
            StepMotionTransform(
            ), apply_probability=0.005
        ))

        tr_transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                         # exclude_labels=(0, 1, 10, 9),
                                         skip_minus_one_selected_label=False,
                                         verbose=False,
                                         ), apply_probability=0.005
        ))
        # tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms
    



    
class nnUNetTrainerCT3DSSLMAENNUNetArtifactAugMonaiDLP256_ClippedZScore_BS4(nnUNetTrainerCT3DSSLMAENNUNetP256_BS96):
    def __init__(self, **kwargs):
        kwargs["batch_size"] = 4
        super().__init__(**kwargs)
        self.data_preprocessor=FullPreprocessing(
            dataset_json=self.dataset_json,
                    plans_manager=self.plans_manager,
                    configuration_manager=self.configuration_manager,
                    normalization_cls=ClippedZScoreNormalization
                    )
        

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        
        dataset_tr = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=tr_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        dataset_val = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        
        return dataset_tr, dataset_val
    

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> AbstractTransform:
        val_transforms = []
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            # ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            # ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0,
                random_crop=False, p_elastic_deform=0,
                p_rotation=0.1,
                rotation=rotation_for_DA, p_scaling=0.1, scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        # if mirror_axes is not None and len(mirror_axes) > 0:
        #     tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(RandomTransform(
            StepMotionTransform(
            ), apply_probability=0.05
        ))

        tr_transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                         # exclude_labels=(0, 1, 10, 9),
                                         skip_minus_one_selected_label=False,
                                         verbose=False,
                                         ), apply_probability=0.05
        ))
        # tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms
    
    
class nnUNetTrainerCT3DSSLMAENNUNetArtifactAugMonaiDLP256_ClippedZScore_BS8(nnUNetTrainerCT3DSSLMAENNUNetP256_BS96):
    def __init__(self, **kwargs):
        kwargs["batch_size"] = 8
        super().__init__(**kwargs)
        self.data_preprocessor=FullPreprocessing(
            dataset_json=self.dataset_json,
                    plans_manager=self.plans_manager,
                    configuration_manager=self.configuration_manager,
                    normalization_cls=ClippedZScoreNormalization
                    )
        

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        
        dataset_tr = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=tr_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        dataset_val = MonaiDatasetNumpyNiiGz(dict_input=None, folder=self.preprocessed_dataset_folder, identifiers=val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                         target_type=self.target_type)
        
        return dataset_tr, dataset_val
    

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> AbstractTransform:
        val_transforms = []
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            # ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            # ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0,
                random_crop=False, p_elastic_deform=0,
                p_rotation=0.1,
                rotation=rotation_for_DA, p_scaling=0.1, scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        # if mirror_axes is not None and len(mirror_axes) > 0:
        #     tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(RandomTransform(
            StepMotionTransform(
            ), apply_probability=0.05
        ))

        tr_transforms.append(RandomTransform(
            RandomArtifactTransform(spacing=self.configuration_manager.spacing,
                                         # exclude_labels=(0, 1, 10, 9),
                                         skip_minus_one_selected_label=False,
                                         verbose=False,
                                         ), apply_probability=0.05
        ))
        # tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms