from deep_utils import SITKUtils
import os
from time import time
from typing import Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.utilities.file_and_folder_operations import join
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.center_crop_dataloader import CenterCropDataLoader
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.indexable_dataloader import IndexableSingleThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.mae_mse_loss import MAEL1LossMSELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.ssl_utils import mask_creation
from torch import autocast
from torch import distributed as dist
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainerMonaiDL
# from monai.transforms import (
#     RandAffined,
#     RandFlipd,
#     EnsureChannelFirstd,
# )


class nnUNetTrainerSSL(nnUNetTrainer):
    def __init__(
            self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda'), num_epochs: int = 1000, initial_lr=0.01,
            batch_size: int = 6, **kwargs,
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs,
                         initial_lr=initial_lr,
                         batch_size=batch_size, **kwargs)
        self.mask_ratio: float = 0.75
        self.mask_block_size: int = 16
        self.whole_volume_masking: bool = False
        self.autocast_dtype = torch.float16
        self.im_output_folder = os.path.join(self.output_folder, "img_log")
        os.makedirs(self.im_output_folder, exist_ok=True)
        self.apply_qualitative_per_n_epochs = None
        self.unpack_segmentation = False
        self.enable_deep_supervision = False
        self.add_skip_connection = False
        self.monai_dl = False

    def initialize(self):
        if not self.was_initialized:
            if self.autocast_dtype == torch.bfloat16:
                self.grad_scaler = None
            super().initialize()
            # SSL trainers may not use all parameters (e.g. decoder skip connections unused
            # in MAE pretraining), so DDP must allow unused parameters.
            # if self.is_ddp and isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
            #     from torch.nn.parallel import DistributedDataParallel as DDP
            #     inner = self.network.module
            #     self.network = DDP(inner, device_ids=[self.local_rank], find_unused_parameters=True)

            # eps=1e-5 is subnormal in fp16 and gets flushed to zero on NVIDIA GPUs,
            # causing division-by-zero in InstanceNorm when masked inputs produce near-zero variance.
            # bf16 shares float32's exponent range so 1e-5 is normal there; skip the patch.
            if self.autocast_dtype == torch.float16:
                _instance_norm_types = (
                    torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d
                )
                for m in self.network.modules():
                    if isinstance(m, _instance_norm_types) and m.eps < 1e-4:
                        m.eps = 1e-4

            self.qualitative_dl = self.get_qualitatvie_dataloader()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss", np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4)
        )
        self.print_to_log_file("val_loss", np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )
        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        is_best = False
        # handle 'best' checkpointing. val_loss smaller than best_ema
        if self._best_ema is None or self.logger.my_fantastic_logging["val_losses"][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging["val_losses"][-1]
            self.print_to_log_file(f"Yayy! New best val loss: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
            is_best = True

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
        return is_best, False

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def _build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        # For SSL this is not required!
        return MAEL1LossMSELoss((1, 1), ddp=False) 

    def train_step(self, batch: dict) -> dict:
        if self.monai_dl:
            data, properties, target = batch
        else:
            data = batch["data"]
        data = data.to(self.device, non_blocking=True)
        if data.isnan().any():
            self.print_to_log_file(f"[WARNING] nan in input, skipping batch: {batch['keys']}")
            return {"loss": np.nan}

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        if self.whole_volume_masking and batch.get("mask") is not None:
            mask = batch["mask"].to(self.device, non_blocking=True)
        else:
            mask = mask_creation(self.batch_size, self.configuration_manager.patch_size, self.mask_ratio,
                                 block_size=self.mask_block_size).to(self.device, non_blocking=True)

        masked_data = data * mask

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(masked_data)
            l = self.loss(output, data, mask)

        if output.isnan().any():
            self.print_to_log_file(f"[WARNING] nan in output, skipping batch: {batch['keys']}")
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": np.nan}

        if torch.isnan(l):
            self.print_to_log_file(f"[WARNING] nan loss, skipping batch: {batch['keys']}")
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": np.nan}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        if self.monai_dl:
            data, properties, target = batch
        else:
            data = batch["data"]
        data = data.to(self.device, non_blocking=True)
        
        if data.isnan().any():
            self.print_to_log_file(f"[WARNING] nan in input, skipping val batch: {batch['keys']}")
            return {"loss": np.nan}
        if self.whole_volume_masking and batch.get("mask") is not None:
            mask = batch["mask"].to(self.device, non_blocking=True)
        else:
            mask = mask_creation(self.batch_size, self.configuration_manager.patch_size, self.mask_ratio,
                                 block_size=self.mask_block_size).to(self.device, non_blocking=True)

        masked_data = data * mask

        with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(masked_data)

            if output.isnan().any():
                self.print_to_log_file(f"[WARNING] nan in output, skipping batch: {batch['keys']}")
            
            
            l = self.loss(output, data, mask)
            if torch.isnan(l):
                self.print_to_log_file(f"[WARNING] nan loss in val batch: {batch['keys']}")
                return {"loss": np.nan}
            
        return {"loss": l.detach().cpu().numpy()}

    @staticmethod
    def log_image_and_reco(img: np.ndarray, reco: np.ndarray, mask: np.ndarray, loss=None, filepath: str = None,
                           rescale_type: str = None, smooth: bool = False) -> None:

        # img, reco = nnUNetTrainerSSL.rescale_images(
        #     img, reco, rescale_type
        # )
        # img = (img - img.min()) / (img.max() - img.min())
        # reco = (reco - reco.min()) / (reco.max() - reco.min())
        # print("Start log_image_and_reco")
        img_mask = img * mask
        reco_masked = img_mask + reco * (1 - mask)

        full_img_min, full_img_max = img.min(), img.max()
        denominator = (full_img_max - full_img_min)
        # if denominator == 0:
        #     v = 10
        img = (img - full_img_min) / denominator
        reco = (reco - full_img_min) / denominator
        reco_masked = (reco_masked - full_img_min) / denominator
        

        _, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))

        img_uint8 = (img * 255.0).astype(np.uint8)
        img_mask_uint8 = (img * mask * 255).astype(np.uint8)
        mask_unint8 = (mask * 255.0).astype(np.uint8)
        reco_uint8 = (reco * 255.0).astype(np.uint8)
        reco_uint8_masked = (reco_masked * 255.0).astype(np.uint8)

        # reco_uint8_masked = img_mask_uint8 + reco_uint8 * (1- mask)
        if smooth:
            import scipy.ndimage as ndimage
            reco_uint8_masked = ndimage.gaussian_filter(reco_uint8, sigma=(1, 1), order=0)

        ax[0].imshow(img_uint8, cmap="gray")
        ax[0].set_title("img", fontsize=15)

        ax[1].imshow(mask_unint8, cmap="gray")
        ax[1].set_title("mask", fontsize=15)

        ax[2].imshow(img_mask_uint8, cmap="gray")
        ax[2].set_title(f"m-img", fontsize=15)

        ax[3].imshow(reco_uint8, cmap="gray")
        ax[3].set_title(f"output", fontsize=15)

        ax[4].imshow(reco_uint8_masked, cmap='gray')
        if loss is not None:
            ax[4].set_title(f"m-output, Loss: {float(loss):.05f}", fontsize=15)
        else:
            ax[4].set_title(f"m-output", fontsize=15)

        for a in ax:
            a.axis(False)
            a.grid(False)

        # plt.title(f"Loss: {float(loss):.05f}")
        # plt.savefig(join(self.im_output_folder, filename))
        if filepath is not None:
            plt.savefig(filepath)
        plt.close()
        # print("End log_image_and_reco")
        
    @staticmethod
    def rescale_images(
            img_arr: torch.Tensor | np.ndarray, recon_arr: torch.Tensor | np.ndarray, rescale_type: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray]:
        if rescale_type == "z-score-reverse":
            mean = img_arr.mean()
            std = img_arr.std()
            img_arr = (img_arr * (max(std, 1e-8))) + mean
            rec_arr = (recon_arr * (max(std, 1e-8))) + mean
        elif rescale_type == "separate":
            full_img_min, full_img_max = img_arr.min(), img_arr.max()
            full_reco_min, full_reco_max = recon_arr.min(), recon_arr.max()
            img_arr = (img_arr - full_img_min) / (full_img_max - full_img_min)
            rec_arr = (recon_arr - full_reco_min) / (full_reco_max - full_reco_min)
        else:
            full_img_min, full_img_max = img_arr.min(), img_arr.max()
            img_arr = (img_arr - full_img_min) / (full_img_max - full_img_min)
            rec_arr = (recon_arr - full_img_min) / (full_img_max - full_img_min)
        return img_arr, rec_arr

    def log_img_slices(self, imgs, recos, masks, losses, batch_id: int, sample_index: int):
        offset = batch_id * self.batch_size
        two_d = len(imgs.shape) == 4
        for i in range(recos.shape[0]):
            img = torch.squeeze(imgs[i])
            rec = torch.squeeze(recos[i])
            msk = torch.squeeze(masks[i])
            loss = torch.squeeze(losses[i])
            if not two_d:
                slice_of_choice = int(img.shape[0] // 2)
                msk = msk[slice_of_choice]
                rec = rec[slice_of_choice]
                img = img[slice_of_choice]
            msk = msk.detach().cpu().float().numpy()
            rec = rec.detach().cpu().float().numpy()
            img = img.detach().cpu().float().numpy()
            if np.min(img) == np.max(img):
                self.print_to_log_file(f"img for {batch_id=} {sample_index=} {i=} is full zero, skipping!")
                continue
            if self.local_rank == 0:
                self.log_image_and_reco(img, rec, msk, loss, filepath=join(self.im_output_folder,
                                                                           f"epoch_{self.current_epoch}_{offset + i}_{sample_index + 1}.png"))

    def log_qualitative_step(self):
        """For each sample in the validation dataloader,"""
        if self.local_rank == 0:
            self.print_to_log_file("[INFO] Started log_qualitative_step")
            # Use the unwrapped model to avoid DDP collective ops on rank-0-only inference
            network = self.network.module if self.is_ddp else self.network
            with torch.no_grad():
                for batch_id in range(min(self.n_quantitative_images, len(self.qualitative_dl))):
                    for sample_index in range(self.n_qualitative_samples):
                        if sample_index == 0:
                            data = self.qualitative_dl[batch_id]["data"]
                        else:
                            data = self.qualitative_dl[batch_id, True]["data"]  # Pick random z-index
                        if isinstance(data, np.ndarray):
                            data = torch.from_numpy(data).float()
                        data = data.to(self.device, non_blocking=True)
                        if self.mask_ratio > 0:
                            mask = mask_creation(
                                1, self.configuration_manager.patch_size, self.mask_ratio,
                                block_size=self.mask_block_size
                                # rng_seed=123 + batch_id
                            ).to(self.device, non_blocking=True)

                            masked_data = data * mask
                        else:
                            masked_data = data

                        with autocast(self.device.type, dtype=self.autocast_dtype, enabled=True) if self.device.type == "cuda" else dummy_context():
                            reconstruction = network(masked_data)

                            l = [
                                self.loss(reconstruction[i: i + 1], data[i: i + 1], mask[i: i + 1])
                                for i in range(reconstruction.shape[0])
                            ]
                            self.log_img_slices(data, reconstruction, mask, l, batch_id, sample_index)
            self.print_to_log_file("[INFO] Ended log_qualitative_step")
    
    def get_qualitatvie_dataloader(self):

        self._set_batch_size_and_oversample()
        deep_supervision_scales = self._get_deep_supervision_scales()
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=None,
            regions=None,
            ignore_label=None
        )
        dl_val = self.get_centercrop_val_dataloader(val_transforms)

        mt_gen_val = IndexableSingleThreadedAugmenter(dl_val, transform=None)
        return mt_gen_val

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            plans_manager=None,
            configuration_manager=None,
    ) -> AbstractTransform:
        val_transforms = []
        # val_transforms.append(NumpyToTensor(["data"], "float"))
        val_transforms = Compose(val_transforms)
        return val_transforms

    @staticmethod
    def get_training_transforms(
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
            plans_manager=None,
            configuration_manager=None,
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
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        # tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_centercrop_val_dataloader(self, val_transforms=None):
        """Returns a centercropped dataloader."""
        _, dataset_val = self.get_tr_and_val_datasets()

        dl_val = CenterCropDataLoader(
            dataset_val,
            1,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            transforms=val_transforms,
            label_manager=None,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            target_type=self.target_type,
            data_preprocessor=self.data_preprocessor
        )
        return dl_val

    def get_dataloaders(self, dl_class=None, dl_class_args=None):
        """
        This is different compared to get_dataloaders in the sense that tr_transforms is different!
        :param dl_class:
        :param dl_class_args:
        :return:
        """
        self.print_to_log_file(f"[INFO] get_dataloaders with dl_class: {dl_class} and dl_class_args: {dl_class_args}")
        tic = time()

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        # deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, None, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None)

        # validation pipeline
        val_transforms = self.get_validation_transforms(None)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling,
                                 target_type=self.target_type,
                                 data_preprocessor=self.data_preprocessor,
                                 whole_volume_masking=self.whole_volume_masking,
                                 mask_ratio=self.mask_ratio,
                                 mask_block_size=self.mask_block_size)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling,
                                  target_type=self.target_type,
                                  data_preprocessor=self.data_preprocessor,
                                  whole_volume_masking=self.whole_volume_masking,
                                  mask_ratio=self.mask_ratio,
                                  mask_block_size=self.mask_block_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        print(f"[INFO] Allowed number of processes: {allowed_num_processes}")
        if dl_class is not None:
            dl_class_args = dl_class_args or dict()
            mt_gen_train = dl_class(dl_tr, **dl_class_args)
            mt_gen_val = dl_class(dl_val, **dl_class_args)
        else:
            if allowed_num_processes == 0 or allowed_num_processes == 1:
                mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
                mt_gen_val = SingleThreadedAugmenter(dl_val, None)
            else:
                mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                            num_processes=allowed_num_processes,
                                                            num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                            pin_memory=self.device.type == 'cuda', wait_time=0.002)
                mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                          transform=None,
                                                          num_processes=max(1, allowed_num_processes // 2),
                                                          num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                          pin_memory=self.device.type == 'cuda',
                                                          wait_time=0.002)
        # let's get this party started
        self.print_to_log_file(f"Starting to get one batch data!")
        output = next(mt_gen_train)
        _ = next(mt_gen_val)

        self.print_to_log_file(f"get_dataloaders took: {time() - tic}, output: {output['data'].shape}")
        return mt_gen_train, mt_gen_val

    def perform_quantitative_validation(self, save_probabilities: bool = False):
        if self.local_rank != 0:
            return
        # Unwrap DDP for rank-0-only inference and suppress the dist.barrier() inside super
        saved_network = self.network
        saved_is_ddp = self.is_ddp
        if self.is_ddp:
            self.network = self.network.module
            self.is_ddp = False
        try:
            super().perform_quantitative_validation(save_probabilities)
        finally:
            self.network = saved_network
            self.is_ddp = saved_is_ddp

    def _get_validation_predictor_settings(self):
        return False, False, 1.0

    def _prepare_validation_sample(self, data, predictor, properties, output_filename_truncated):
        saving_spacing = (
            self.configuration_manager.spacing[::-1]
            if -1 not in self.configuration_manager.spacing
            else properties['spacing']
        )
        if hasattr(self.network, "add_mask"):
            self.network.add_mask = True
            self.print_to_log_file("performing quantitative validation for ssl: add_mask is set to True")
        elif self.mask_ratio:
            # creating mask and multiplying it by the whole data is wrong and makes stupid results!
            # It should be patch wise!
            SITKUtils.save_sample(
                output_filename_truncated + "_before_resample_data.nii.gz",
                input_array=data.detach().cpu().numpy()[0],
                spacing=saving_spacing,
                origin=properties['sitk_stuff']['origin'],
                direction=properties['sitk_stuff']['direction']
            )
            # Delegate masking to the predictor so it applies a fresh patch-sized mask inside
            # each sliding-window step rather than a single global mask on the full volume.
            predictor.add_mask = True
            predictor.return_mask = True
            predictor.mask_ratio = self.mask_ratio
            predictor.block_size = self.mask_block_size
        else:
            raise ValueError("mask_ratio is not set and the network has no add_mask attribute!")
        return data, saving_spacing

    def _save_validation_extras(self, prediction, pred_mask, properties, output_filename_truncated, saving_spacing):
        SITKUtils.save_sample(
            output_filename_truncated + "_before_resample_pred.nii.gz",
            input_array=prediction.numpy()[0].astype(np.float32),
            spacing=saving_spacing,
            origin=properties['sitk_stuff']['origin'],
            direction=properties['sitk_stuff']['direction']
        )
        if pred_mask is not None:
            SITKUtils.save_sample(
                output_filename_truncated + "_before_resample_mask.nii.gz",
                input_array=pred_mask.numpy()[0].astype(np.float32),
                spacing=saving_spacing,
                origin=properties['sitk_stuff']['origin'],
                direction=properties['sitk_stuff']['direction']
            )

    def _finalize_validation(self):
        if hasattr(self.network, "add_mask"):
            self.network.add_mask = False

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


