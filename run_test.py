from os.path import join
import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_name", default="Dataset005_mri_fat")
parser.add_argument("--model_name", default="M2Net")
parser.add_argument("--base_dir", default="NNUNET_OUTPUT")
parser.add_argument("--device", type=int, default=1)

trainer_name = dict(
    nnUNetTrainer=""
)
if __name__ == '__main__':
    args = parser.parse_args()
    nnunet_raw = f"{args.base_dir}/nnunet_raw"
    nnunet_results = f"{args.base_dir}/nnunet_results"

    trainer__ = "nnUNetTrainer" + trainer_name.get(args.model_name, args.model_name)
    model_path = join(nnunet_results, args.dataset_name, trainer__ + '__nnUNetPlans__2d')
    input_path = join(nnunet_raw, args.dataset_name, "imagesTs")
    output_path = join(nnunet_raw, args.dataset_name, f"imagesTs_{args.model_name}_Pred")

    res = os.system(
        f"python code_inference.py --device cuda:1 --model_path  {model_path} --input {input_path} --output {output_path} --ext {'.png' if args.dataset_name == 'Dataset032_NeurlPSCell' else '.gz'}")
    if res != 0:
        print(
            f"[Error] Couldn't do {args.model_name} on {args.dataset_name} with command: \n python code_inference.py --device cuda:0 --model_path  {model_path} --input {input_path} --output {output_path} --ext {'.png' if args.dataset_name == 'Dataset032_NeurlPSCell' else '.gz'}")
        sys.exit(1)
    input_lbl_path = join(nnunet_raw, args.dataset_name, "labelsTs")
    summary_path = join(model_path, 'test_summary.json')
    if args.dataset_name != 'Dataset032_NeurlPSCell':
        res = os.system(
            f"CUDA_VISIBLE_DEVICES=1 nnUNetv2_evaluate_folder  {input_lbl_path} {output_path} -djfile {join(model_path, 'dataset.json')} -pfile {join(model_path, 'plans.json')} -o {summary_path}")
    else:
        res = os.system(
            f"python compute_cell_metric.py --gt_path {input_lbl_path} --seg_path {output_path} --save_path {summary_path}"
        )
    if res != 0:
        print(f"[ERROR] Couldn't evaluate {args.model_name} and {args.dataset_name}")
    print(f"[INFO] FINISHED {args.model_name} and {args.dataset_name}")
