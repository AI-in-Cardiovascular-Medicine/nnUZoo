import os
import socket
from typing import Union, Optional
from os.path import exists

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from torch.backends import cudnn
from deep_utils import StringUtils


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int | str,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda'),
                          num_epochs: int = 1000, initial_lr: float = 1e-2,
                          up_sample_type: str = 'convtranspose',
pretrained_weights:str = None,
                          **kwargs):
    # load nnunet class and do sanity checks
    # print(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"))
    nnunet_trainer = recursive_find_trainer_class_by_name(trainer_name)
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                      'nnUNetTrainer'

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    if exists(plans_file):
        plans = load_json(plans_file)
    else:
        plans_file = join(preprocessed_dataset_folder_base, 'nnUNetResEncUNetLPlans.json')
        plans = load_json(plans_file)
    dataset_json = load_json(join(nnUNet_raw, maybe_convert_to_dataset_name(dataset_name_or_id), 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device,
                                    num_epochs=num_epochs, initial_lr=initial_lr,
                                    up_sample_type=up_sample_type,
                                    pretrained_weights=pretrained_weights,
                                    preprocessed_dataset_folder_base=preprocessed_dataset_folder_base,
                                    **kwargs)
    # print(nnunet_trainer._get_deep_supervision_scales())
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None, only_encoder: bool = False, strict_in_missing_weights: bool = False):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                  f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            if hasattr(nnunet_trainer, "load_pretrained_weights"):
                nnunet_trainer.load_pretrained_weights(pretrained_weights_file)
            else:
                load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True,
                                        only_encoder=only_encoder, logger_func=nnunet_trainer.print_to_log_file,
                                        strict_in_missing_weights=strict_in_missing_weights)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                           use_compressed)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda'),
                 num_epochs: int = 1000,
                 skip_val: bool = False,
                 initial_lr: float = 1e-2,
                 only_encoder: bool = False,
                 strict_in_missing_weights: bool = True,
                 up_sample_type: str = "convtranspose",
                 batch_size: int = None,
                 **kwargs
                 ):
    if isinstance(fold, str):
        if fold.startswith("get_one_fold"):
            pass
        elif fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    else:
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device,
                                               num_epochs=num_epochs, initial_lr=initial_lr,
                                               up_sample_type=up_sample_type,
                                               batch_size=batch_size,
                                               pretrained_weights=pretrained_weights,
                                               **kwargs)
        nnunet_trainer.print_to_log_file(f"args:{dict(num_epochs=num_epochs, batch_size=batch_size, 
                                                      initial_lr=initial_lr, up_sample_type=up_sample_type,
                                                      pretrained_weights=pretrained_weights,
                                                      fold=fold,use_compressed_data=use_compressed_data,
                                                      val_with_best=val_with_best,num_gpus=num_gpus,
                                                      only_run_validation=only_run_validation,
                                                      continue_training=continue_training,trainer_class_name=trainer_class_name,
                                                      dataset_name_or_id=dataset_name_or_id,
                                                      )}")
        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
                continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights,
                              only_encoder=only_encoder, strict_in_missing_weights=strict_in_missing_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if skip_val:
            nnunet_trainer.print_to_log_file("[INFO] Validation is set to False!")
        else:
            if val_with_best or nnunet_trainer.target_type in ['classification']:
                nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
                nnunet_trainer.print_to_log_file("Loading best checkpoint")
            nnunet_trainer.perform_actual_validation(export_validation_probabilities)
            


def run_unpacking(dataset_name_or_id: Union[str, int],
                  configuration: str, fold: Union[int, str],
                  trainer_class_name: str = 'nnUNetTrainer',
                  plans_identifier: str = 'nnUNetPlans',
                  pretrained_weights: Optional[str] = None,
                  num_gpus: int = 1,
                  use_compressed_data: bool = False,
                  export_validation_probabilities: bool = False,
                  continue_training: bool = False,
                  only_run_validation: bool = False,
                  disable_checkpointing: bool = False,
                  val_with_best: bool = False,
                  device: torch.device = torch.device('cuda'),
                  num_epochs: int = 1000,
                  skip_val: bool = False,
                  initial_lr: float = 1e-2,
                  only_encoder: bool = False,
                  up_sample_type: str = "convtranspose",
                  np: int = default_num_processes,
                  **kwargs
                  ):
    # if isinstance(fold, str):
    #     if fold != 'all':
    #         try:
    #             fold = int(fold)
    #         except ValueError as e:
    #             print(
    #                 f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
    #             raise e

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                           plans_identifier, use_compressed_data, device=device,
                                           num_epochs=num_epochs, initial_lr=initial_lr,
                                           up_sample_type=up_sample_type, **kwargs)

    unpack_dataset(nnunet_trainer.preprocessed_dataset_folder, unpack_segmentation=True,
                   overwrite_existing=False,
                   num_processes=default_num_processes,
                   )


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('-initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate')
    parser.add_argument('-batch_size', type=int, default=None, required=False,
                        help='Batch Size, Default is to use the one in the processing!')
    parser.add_argument('-num_epochs', type=int, default=1000, required=False,
                        help='Number of training epochs')
    parser.add_argument('--decoder_warmup_epochs', type=int, default=50, required=False,
                        help='Number of training epochs')
    parser.add_argument('--encoder_warmup_epochs', type=int, default=25, required=False,
                        help='Number of training epochs')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=50, required=False,
                        help='Number of training epochs')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument("--up_sample_type", default="convtranspose", required=False, help="")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--only_encoder', action='store_true', required=False,
                        help='[OPTIONAL] Only encoder is loaded in pretraining section!')
    parser.add_argument('--not_strict_in_missing_weights', action='store_true', required=False,
                        help='[OPTIONAL] If you have added new layers and want the model don"t become strict on it because it"s missing from your pretrianed weights, set this to True!')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--skip_val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        try: 
            torch.set_num_threads(multiprocessing.cpu_count())
        except Exception as e:
            StringUtils.print("Error in torch.set_num_threads", e)

        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        try: 
            torch.set_num_threads(1)
        except Exception as e:
            StringUtils.print("Error in torch.set_num_threads(1)", e)
        try: 
            torch.set_num_interop_threads(1)
        except Exception as e:
            StringUtils.print("Error in torch.set_num_interop_threads(1)", e)

        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    print(f"[INFO] Training: {args}")
    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.use_compressed, args.npz, args.c, args.val or args.val_best,
                 args.disable_checkpointing, args.val_best,
                 device=device,
                 num_epochs=args.num_epochs,
                 skip_val=args.skip_val,
                 initial_lr=args.initial_lr,
                 only_encoder=args.only_encoder,
                 strict_in_missing_weights=not args.not_strict_in_missing_weights,
                 encoder_warmup_epochs=args.encoder_warmup_epochs,
                 decoder_warmup_epochs=args.decoder_warmup_epochs,
                 freeze_encoder_epochs=args.freeze_encoder_epochs,
                 up_sample_type=args.up_sample_type,
                 batch_size=args.batch_size)


def run_unpacking_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help='number of preprocessor used for unpacking')
    parser.add_argument('-initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate')
    parser.add_argument('-num_epochs', type=int, default=250, required=False,
                        help='Number of training epochs')
    parser.add_argument('--decoder_warmup_epochs', type=int, default=50, required=False,
                        help='Number of training epochs')
    parser.add_argument('--encoder_warmup_epochs', type=int, default=25, required=False,
                        help='Number of training epochs')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=50, required=False,
                        help='Number of training epochs')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument("--up_sample_type", default="convtranspose", required=False, help="")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--only_encoder', action='store_true', required=False,
                        help='[OPTIONAL] Only encoder is loaded in pretraining section!')
    parser.add_argument('--not_strict_in_missing_weights', action='store_true', required=False,
                        help='[OPTIONAL] If you have added new layers and want the model don"t become strict on it because it"s missing from your pretrianed weights, set this to True!')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--skip_val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    print(f"[INFO] Training: {args}")
    run_unpacking(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                  args.num_gpus, args.use_compressed, args.npz, args.c, args.val or args.val_best,
                  args.disable_checkpointing, args.val_best,
                  device=device, num_epochs=args.num_epochs, skip_val=args.skip_val,
                  initial_lr=args.initial_lr, only_encoder=args.only_encoder,
                  strict_in_missing_weights=not args.not_strict_in_missing_weights,
                  encoder_warmup_epochs=args.encoder_warmup_epochs,
                  decoder_warmup_epochs=args.decoder_warmup_epochs,
                  freeze_encoder_epochs=args.freeze_encoder_epochs,
                  up_sample_type=args.up_sample_type, np=args.np)


if __name__ == '__main__':
    run_training_entry()
