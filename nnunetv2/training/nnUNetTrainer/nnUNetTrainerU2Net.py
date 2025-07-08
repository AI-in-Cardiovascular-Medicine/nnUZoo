from typing import Union, List, Literal, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary

from nnunetv2.nets.u2net import get_u2net_from_plans
from nnunetv2.nets.u2net import get_u2netp_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager


class nnUNetTrainerU2Net(nnUNetTrainer):
    """ Swin-UMamba """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         **kwargs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = -1  # training from scratch
        self.early_stop_epoch = 10

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
            *,
            up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
            configuration_manager: ConfigurationManager = None
    ) -> nn.Module:

        model = get_u2net_from_plans(
            num_output_channels,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
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

    # def on_train_epoch_start(self):
    #     # freeze the encoder if the epoch is less than 10
    #     if self.current_epoch < self.freeze_encoder_epochs:
    #         self.print_to_log_file("Freezing the encoder")
    #         if self.is_ddp:
    #             self.network.module.freeze_encoder()
    #         else:
    #             self.network.freeze_encoder()
    #     else:
    #         self.print_to_log_file("Unfreezing the encoder")
    #         if self.is_ddp:
    #             self.network.module.unfreeze_encoder()
    #         else:
    #             self.network.unfreeze_encoder()
    #     super().on_train_epoch_start()

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled


class nnUNetTrainerU2NetP(nnUNetTrainerU2Net):
    """ Swin-UMamba """

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True,
            *,
            up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
            configuration_manager: ConfigurationManager = None

    ) -> nn.Module:
        model = get_u2netp_from_plans(
            num_output_channels,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model
