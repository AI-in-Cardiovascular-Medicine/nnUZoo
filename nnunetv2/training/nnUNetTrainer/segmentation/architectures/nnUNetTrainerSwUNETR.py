from typing import Literal

import torch
from monai.networks.nets import SwinUNETR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class nnUNetTrainerSwUNETR(nnUNetTrainer):
    """ Swin-UNETR """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device,
                         num_epochs=num_epochs, **kwargs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False

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
        network = SwinUNETR(
            # img_size=configuration_manager.patch_size[0],
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=len(configuration_manager.patch_size),
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        return network

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
