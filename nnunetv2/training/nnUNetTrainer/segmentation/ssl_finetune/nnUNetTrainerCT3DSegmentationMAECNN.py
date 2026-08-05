from typing import Literal
import types

import torch
from nnunetv2.training.lr_scheduler.polylr import ContinuedPolyLRSchedulerWithWarmup
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer



def freeze_encoder_nnunet(self):
    for name, param in self.encoder.named_parameters():
        param.requires_grad = False


def unfreeze_encoder_nnunet(self):
    for param in self.encoder.parameters():
        param.requires_grad = True


## nnUnet format:
class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
            num_epochs: int = 1000,
            initial_lr: float = 1e-2,
            up_sample_type: Literal['convtranspose', 'trilinear', "nearest"] = 'convtranspose',
            batch_size: int = None,
            **kwargs
    ):
        batch_size = batch_size or 2
        initial_lr = 1e-3
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
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         initial_lr=initial_lr, up_sample_type=up_sample_type, batch_size=batch_size, **kwargs)
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = 25
        self.warmup_epochs = 5
        self.initial_lr = 1e-3
        self.warmup_lr = 1e-4
        self.final_lr = 1e-6

        self.weight_decay = 5e-2
        self.percent = 1
        self.autocast_dtype = torch.bfloat16

    def do_split(self):
        if self.percent == 1:
            return super().do_split()
        tr_keys, val_keys = super().do_split()
        percent = self.percent
        n_t = max(1, round(len(tr_keys) * percent))
        n_v = max(1, round(len(val_keys) * percent))
        tr_keys, val_keys = tr_keys[:n_t], val_keys[:n_v]
        self.print_to_log_file("This split has %d training and %d validation cases."
                               % (len(tr_keys), len(val_keys)))
        return tr_keys, val_keys

    def initialize(self):
        super().initialize()
        self.network.freeze_encoder = types.MethodType(freeze_encoder_nnunet, self.network)
        self.network.unfreeze_encoder = types.MethodType(unfreeze_encoder_nnunet, self.network)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ContinuedPolyLRSchedulerWithWarmup(optimizer, self.initial_lr,
                                                          total_epochs=self.num_epochs,
                                                          warmup_epochs=self.warmup_epochs,
                                                          final_lr=self.final_lr,
                                                          warmup_lr=self.warmup_lr,
                                                          initial_lr=self.initial_lr)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.freeze_encoder_epochs != -1:
            if self.current_epoch < self.freeze_encoder_epochs:
                self.print_to_log_file("Freezing the encoder")
                if self.is_ddp:
                    self.network.module.freeze_encoder()
                else:
                    self.network.freeze_encoder()
            else:
                self.print_to_log_file("Unfreezing the encoder")
                if self.is_ddp:
                    self.network.module.unfreeze_encoder()
                else:
                    self.network.unfreeze_encoder()
        super().on_train_epoch_start()


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
            num_epochs: int = 1000,
            initial_lr: float = 1e-2,
            up_sample_type: Literal['convtranspose', 'trilinear', "nearest"] = 'convtranspose',
            batch_size: int = None,
            **kwargs
    ):
        batch_size = batch_size or 2
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
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         initial_lr=initial_lr, up_sample_type=up_sample_type, batch_size=batch_size, **kwargs)
        self.percent = 1
        self.autocast_dtype = torch.bfloat16


    def do_split(self):
        if self.percent == 1:
            return super().do_split()
        tr_keys, val_keys = super().do_split()
        percent = self.percent
        n_t = max(1, round(len(tr_keys) * percent))
        n_v = max(1, round(len(val_keys) * percent))
        tr_keys, val_keys = tr_keys[:n_t], val_keys[:n_v]

        self.print_to_log_file("This split has %d training and %d validation cases."
                               % (len(tr_keys), len(val_keys)))
        return tr_keys, val_keys


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_01Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder_epochs = 10
        self.percent = 0.01


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_05Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder_epochs = 10
        self.percent = 0.05


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_10Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder_epochs = 10
        self.percent = 0.1


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_20Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.2


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_50Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.5


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_75Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.75


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_100Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    pass

###### Special Case ###### 
class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_NoMirror_Pretrain_100Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

######s Special Case ######


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_01Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.01


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_05Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.05


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_10Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.1


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_20Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.2


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_50Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.5


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_75Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent = 0.75


class nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_100Percent(
    nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2):
    pass
        