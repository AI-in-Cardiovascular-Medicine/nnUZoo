import random
import time
import types
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.optim import AdamW
from nnunetv2.training.lr_scheduler.polylr import ContinuedPolyLRSchedulerWithWarmup
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn


def freeze_encoder(self):
    for name, param in self.encoder.named_parameters():
        param.requires_grad = False


def unfreeze_encoder(self):
    for param in self.encoder.parameters():
        param.requires_grad = True


def check_validity(self, state: bool):
    param_index = random.randint(0, len(list(self.encoder.parameters())) - 1)
    for idx, param in enumerate(self.encoder.parameters()):
        if idx == param_index:
            if state != param.requires_grad:
                print(f"[ERROR] Encoder param.requires_grad should be {state} but it is {param.requires_grad}")
            else:
                print(f" Encoder param.requires_grad should be {state} and it is {param.requires_grad}")


class nnUNetTrainerFineTuneDoubleWarmUP(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250,
                 initial_lr: float = 1e-3, decoder_warmup_epochs: int = 50, encoder_warmup_epochs: int = 25,
                 freeze_encoder_epochs: int = 50):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs,
                         initial_lr=initial_lr,
                         encoder_warmup_epochs=encoder_warmup_epochs,
                         decoder_warmup_epochs=decoder_warmup_epochs,
                         freeze_encoder_epochs=freeze_encoder_epochs,
                         )
        self.initial_lr = initial_lr
        self.weight_decay = 5e-2
        # self.enable_deep_supervision = True
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.decoder_warmup_epochs = decoder_warmup_epochs
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.early_stop_epoch = 350

        self.unfreeze_encoder = False

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True,
                                   # pretrain_path:str = ''
                                   ) -> nn.Module:
        model = nnUNetTrainer.build_network_architecture(plans_manager,
                                                         dataset_json,
                                                         configuration_manager,
                                                         num_input_channels,
                                                         enable_deep_supervision)

        model.freeze_encoder = types.MethodType(freeze_encoder, model)
        model.unfreeze_encoder = types.MethodType(unfreeze_encoder, model)
        model.check_validity = types.MethodType(check_validity, model)
        return model

    def on_train_epoch_start(self):
        # freeze the encoder if the epoch is less than 10
        if self.freeze_encoder_epochs != -1:
            if self.current_epoch < self.freeze_encoder_epochs:
                self.print_to_log_file("Freezing the encoder")
                if self.is_ddp:
                    self.network.module.freeze_encoder()
                else:
                    self.network.freeze_encoder()
                self.network.check_validity(False)
            elif self.current_epoch >= self.freeze_encoder_epochs:
                if not self.unfreeze_encoder:
                    self.configure_scheduler()

                self.unfreeze_encoder = True
                tic = time.time()
                if self.is_ddp:
                    self.network.module.unfreeze_encoder()
                else:
                    self.network.unfreeze_encoder()
                self.network.check_validity(True)

                elapsed = time.time() - tic
                self.print_to_log_file(f"Unfreezing the encoder, took: {elapsed}")
        super().on_train_epoch_start()

    def configure_scheduler(self):
        total_epochs = self.num_epochs
        scheduler = ContinuedPolyLRSchedulerWithWarmup(
            self.optimizer,
            start_epoch=self.freeze_encoder_epochs,
            initial_lr=self.initial_lr,
            warmup_lr=1e-5,
            final_lr=1e-5,
            total_epochs=total_epochs,
            warmup_epochs=self.decoder_warmup_epochs,
        )
        self.print_to_log_file(
            f"Setting Encoder-Decoder warmup scheduler {scheduler} with start_epoch: 0, total_epochs: {total_epochs}")

        self.lr_scheduler = scheduler

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        # Decoder Warmup
        scheduler = ContinuedPolyLRSchedulerWithWarmup(
            optimizer,
            start_epoch=0,
            initial_lr=self.initial_lr,
            warmup_lr=1e-5,
            final_lr=1e-5,
            total_epochs=self.freeze_encoder_epochs,
            warmup_epochs=self.encoder_warmup_epochs,
        )

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
