"""
This is a handy code for training the models
"""
import os
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument("--dataset_name", default="Dataset005_mri_fat")
parser.add_argument("--base_dir", default="NNUNET_OUTPUT")
parser.add_argument("--tr", default="nnUNetTrainer")
parser.add_argument("--model", default="2d")
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=250)
parser.add_argument("--val", action="store_true")
parser.add_argument("--val_best", action="store_true")
parser.add_argument("--skip_val", action="store_true")
parser.add_argument("--c", action="store_true")


args = parser.parse_args()
if __name__ == '__main__':

    nnunet_raw = f"{args.base_dir}/nnunet_raw"
    nnunet_results = f"{args.base_dir}/nnunet_results_time"
    nnunet_preprocessed = f"{args.base_dir}/nnunet_preprocessed"

    os.makedirs(nnunet_results, exist_ok=True)
    os.environ['nnUNet_raw'] = nnunet_raw
    os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
    os.environ['nnUNet_results'] = nnunet_results

    command = f"CUDA_VISIBLE_DEVICES={args.device} nnUNetv2_train {args.dataset_name} {args.model} 0 -tr {args.tr} {'--val' if args.val else ''} {'--skip_val' if args.skip_val else ''}  {'--val_best' if args.val_best else ''} {'--c' if args.c else ''} -num_epochs {args.num_epochs}"
    os.system(command)
