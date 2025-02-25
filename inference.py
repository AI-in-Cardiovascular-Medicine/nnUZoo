import os
import sys
import time
from os.path import join, split
from deep_utils import DirUtils, NIBUtils, StringUtils
from tqdm import tqdm
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from argparse import ArgumentParser


# Get file paths
def get_files(input_dir: str, processed_files):
    patients = DirUtils.list_dir_full_path(input_dir, only_directories=True)
    delayed_jobs = []
    for patient in tqdm(patients):
        # print("patient",patient)
        for patient_acquizition in DirUtils.list_dir_full_path(patient, only_directories=True):
            for filepath in DirUtils.list_dir_full_path(patient_acquizition, interest_extensions=".gz"):
                if split(filepath)[-1].replace(".nii.gz", "") in processed_files:
                    continue
                delayed_jobs.append(filepath)
    return delayed_jobs


parser = ArgumentParser()
parser.add_argument("--model_path", type=str,
                    default="./NNUNET_OUTPUT/nnunet_results/Dataset034_acdc/nnUNetTrainerM2Net__nnUNetPlans__2d")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--base_dir", default="./NNUNET_OUTPUT")
parser.add_argument("--input", default="./NNUNET_OUTPUT/nnunet_raw/Dataset034_acdc/imagesTs")
parser.add_argument("--output", default="./NNUNET_OUTPUT/nnunet_raw/Dataset034_acdc/imagesTs_M2Net_Pred")
parser.add_argument("--input_val", default=None)
parser.add_argument("--input_fold", default=0, type=int)
parser.add_argument("--remove", action="store_true", help="If set to True removes the output")
parser.add_argument("--reverse", action="store_true", help="Start from the end")
parser.add_argument("--n", default=None, type=int, help="Number of samples to be processed")
parser.add_argument("--ext", default=".gz", type=str)

args = parser.parse_args()

if __name__ == '__main__':

    nnunet_raw = f"{args.base_dir}/nnunet_raw"
    nnunet_results = f"{args.base_dir}/nnunet_results"
    nnunet_preprocessed = f"{args.base_dir}/nnunet_preprocessed"

    os.environ['nnUNet_raw'] = nnunet_raw
    os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
    os.environ['nnUNet_results'] = nnunet_results

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        args.model_path,
        use_folds=(0,),
        # use_folds=(1, 2),
        checkpoint_name='checkpoint_best.pth',
    )

    DirUtils.remove_create(args.output, remove=args.remove)
    remaining_files = DirUtils.list_dir_full_path(args.input,
                                                  only_directories=False,
                                                  get_full_path=True,
                                                  interest_extensions=args.ext)

    processed_files = DirUtils.list_dir_full_path(args.output,
                                                  only_directories=False,
                                                  get_full_path=False,
                                                  interest_extensions=args.ext)

    if len(remaining_files) == 0:
        print("[WARNING] All files were already processed")
        sys.exit(0)
    output_remain = [(StringUtils.right_replace(join(args.output, split(filepath)[-1]), "_0000", "", 1), filepath) for
                     filepath in remaining_files if
                     StringUtils.right_replace(split(filepath)[-1], "_0000", "", 1) not in processed_files]
    if len(output_remain):
        output_files, remaining_files = list(zip(*output_remain))
    else:
        print("[WARNING] All files were already processed")
        sys.exit(0)
    print(output_files)
    output_files = output_files[:args.n]
    remaining_files = remaining_files[:args.n]
    if args.reverse:
        output_files = output_files[::-1]
        remaining_files = remaining_files[::-1]
    tic = time.time()
    if len(output_files) != 0:
        predictor.predict_from_files(
            [[item] for item in remaining_files],
            output_files,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0)
        print(f"Processed {len(output_files)} files in {time.time() - tic} seconds!")
    else:
        print("[WARNING] All files were already processed")
