from batchgenerators.utilities.file_and_folder_operations import join
try:
    from nnunetv2.nets.mamba_nd2net import get_mamband2net_from_plans
except:
    pass
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary


class nnUNetTrainerMambaND2Net(nnUNetTrainer):
    """ Swin-UMamba """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        # self.enable_deep_supervision = True
        self.freeze_encoder_epochs = -1  # training from scratch
        # self.early_stop_epoch = 350
    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True,
            use_pretrain: bool = True,
    ) -> nn.Module:
        model = get_mamband2net_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
            small_mode=False
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            spatial_dims = len(self.configuration_manager.patch_size)
            deep_supervision_scales = [
                [1.0] * spatial_dims,
                [1.0] * spatial_dims,
                [0.5] * spatial_dims,
                [0.25] * spatial_dims,
                [0.125] * spatial_dims,
                [0.0625] * spatial_dims,
                [0.03125] * spatial_dims
            ]
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

    def on_epoch_end(self):
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch}.pth'))
        super().on_epoch_end()

    def on_train_epoch_start(self):
        # freeze the encoder if the epoch is less than 10
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

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}


class nnUNetTrainerMambaND2NetP(nnUNetTrainerMambaND2Net):
    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True,
            use_pretrain: bool = True,
    ) -> nn.Module:
        model = get_mamband2net_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_pretrain=False,
            small_mode=True
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model
