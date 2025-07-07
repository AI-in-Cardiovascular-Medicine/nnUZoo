from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torchinfo import summary
import torch
from torch.optim import AdamW
from torch import nn

from monai.networks.nets import UNETR


class nnUNetTrainerUNETR(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 250):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs)
        original_patch_size = self.configuration_manager.patch_size
        self.enable_deep_supervision = False
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            ## 16 is ViT's fixed patch size
            if (original_patch_size[i] / 16) < 1 or ((original_patch_size[i] / 16) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 16 + 0.5) * 16
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.print_to_log_file("Patch size changed from {} to {}".format(original_patch_size, new_patch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size

        self.initial_lr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = UNETR(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            img_size=configuration_manager.patch_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=len(configuration_manager.patch_size),
            qkv_bias=False,
            save_attn=False,
        )
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)
        return model

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

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerUNETRFineTune(nnUNetTrainerUNETR):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        pretrained_path = "/home/aicvi/projects/PET_FM/runs_pet/best_model.pt"
        label_manager = plans_manager.get_label_manager(dataset_json)

        model = UNETR(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            img_size=configuration_manager.patch_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=len(configuration_manager.patch_size),
            qkv_bias=False,
            save_attn=False,
        )

        # Load ViT backbone weights into UNETR
        use_pretrained = True
        if use_pretrained is True:
            print("Loading Weights from the Path {}".format(pretrained_path))
            vit_dict = torch.load(pretrained_path, map_location="cpu")
            vit_weights = vit_dict["state_dict"]

            # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
            # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
            # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
            # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
            model_dict = model.vit.state_dict()

            vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
            model_dict.update(vit_weights)
            model.vit.load_state_dict(model_dict)
            del model_dict, vit_weights, vit_dict
            print("Pretrained Weights Successfuly Loaded !")

        elif use_pretrained is False:
            print("No weights were loaded, all weights being used are randomly initialized!")
        summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)
        return model