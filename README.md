# nnUZoo

<img src="assets/main_img.jpg">


## Contents


## Preprocessing

nnUZoo keeps the nnU-Net v2 data layout and CLI, so any dataset that works with nnU-Net works here
without changes.

### 1. Set the environment variables

Three variables tell nnUZoo where to read and write ([nnunetv2/paths.py](nnunetv2/paths.py)):

```bash
export nnUNet_raw="/path/to/nnUNet_raw"                   # input datasets
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed" # plans + preprocessed data
export nnUNet_results="/path/to/nnUNet_results"           # checkpoints, logs, predictions
```

Add them to your `~/.bashrc` to make them permanent. Every command below fails (or warns) if they are
not set.

### 2. Organize the raw dataset

```
nnUNet_raw
└── Dataset998_mm_whs
    ├── dataset.json
    ├── imagesTr
    │   ├── ct_train_1001__0000.nii.gz     # CASE_XXXX.ext, XXXX = channel/modality index
    │   └── ...
    ├── labelsTr
    │   ├── ct_train_1001_.nii.gz          # CASE.ext, no channel index
    │   └── ...
    ├── imagesTs                      # optional, for the held-out test set
    │   └── ...
    └── labelsTs                      # optional, only needed to evaluate the test set
        └── ...
```

The folder name must follow `DatasetXXX_Name`, where `XXX` are exactly three digits: that ID is what
you pass to every command with `-d`. Images carry a four-digit channel suffix, labels do not, and the
file ending must be identical for images and labels.

### 3. Write `dataset.json`

```json
{
  "channel_names": { "0": "CT" },
  "labels": {
    "background": 0,
    "Coronary Artery": 1,
    "LVM": 2,
    "LA": 3,
    "LV": 4,
    "RA": 5,
    "RV": 6,
    "Aorta": 7,
    "Pulmonary Arteries": 8,
    "Precardial Fat": 9,
    "Epicardial Fat": 10,
    "Pulmonary Vein": 11,
    "Superior Vena Cava": 12,
    "Inferior Vena Cava": 13,
    "LA Appendage": 14
  },
  "numTraining": 17,
  "file_ending": ".nii.gz"
}
```
* Note: Three cases from MMWHS are taken for test set.
* Note: For normalization technique check [normalization section](https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo/blob/main/nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py#L19).

| Key | Required | Notes |
|:----|:---------|:------|
| `channel_names` | yes | Index → name. The name drives the normalization scheme (e.g. `CT` triggers CT intensity normalization) |
| `labels` | yes | Name → value; `background` must be `0` and values must be consecutive. A tuple of values defines a region for region-based training |
| `numTraining` | yes | Number of training cases; used as a sanity check |
| `file_ending` | yes | e.g. `.nii.gz`, `.png`, `.tif` |
| `regions_class_order` | no | Required for region-based training |
| `overwrite_image_reader_writer` | no | Force a specific reader/writer class, e.g. `Tiff3DIO` |
| `target_type` | no | `segmentation` (default) or `ssl` — selects how the data loaders build the target |

You can generate the file programmatically with
`generate_dataset_json()` from [nnunetv2/dataset_conversion/generate_dataset_json.py](nnunetv2/dataset_conversion/generate_dataset_json.py).

### 4. Run fingerprinting, planning and preprocessing

One command does all three steps:

```bash
nnUNetv2_plan_and_preprocess -d 998 -c 3d_fullres --verify_dataset_integrity -pl nnUNetPlannerResEncL
```

Run `--verify_dataset_integrity` at least once per dataset — it catches geometry mismatches, missing
labels and wrong file endings before they cost you a training run.

**Dataset, configurations and workers**

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `-d` | — | **Required.** One or more dataset IDs, e.g. `-d 998 999` |
| `-c` | `2d 3d_fullres 3d_lowres` | Configurations to preprocess. Configurations that a dataset does not support are skipped |
| `-np` | `8` for `2d` / `3d_fullres` / `3d_lowres`, `4` for anything else | Worker processes per configuration. Pass one number for all configurations, or one per entry of `-c`. Resampling is RAM hungry — lower this if memory fills up |
| `-npfp` | `8` | Worker processes for fingerprint extraction |
| `--verbose` | off | Verbose output, no progress bar; recommended on clusters |

**Classes and plans**

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `-fpe` | `DatasetFingerprintExtractor` | Fingerprint extractor class. Only the default ships with the repo; a custom one must live under `nnunetv2.experiment_planning` |
| `-pl` | `ExperimentPlanner` | Planner class; use `ResEncUNetPlanner` for the residual-encoder presets |
| `-gpu_memory_target` | `None` (= the planner's own default, 8 GB) | Target VRAM in GB. Drives patch and batch size — changing it changes results |
| `-preprocessor_name` | `DefaultPreprocessor` | Custom preprocessor class from `nnunetv2.preprocessing` |
| `-overwrite_target_spacing` | `None` | Custom target spacing for `3d_fullres` and `3d_cascade_fullres`, as three numbers. Passing `-1 -1 -1` skips spacing planning and writes `-1` as the spacing of every configuration, i.e. "keep the native spacing" — meant to be combined with `--use_nii` |
| `-overwrite_plans_name` | `None` (= `nnUNetPlans`) | Name of the generated plans file. Use a custom name whenever you change any of the four options above, then pass it to training with `-p`. The plans identifier is handed to the preprocessing step automatically, so you only give it once |
| `--suppress_transpose` | off | Keep the original axis order instead of transposing to put the low-resolution axis first. Use when the z-spacing is not the coarsest one and the dimension order is already correct |

**Which stages to run**

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `--verify_dataset_integrity` | off | Check the dataset before doing any work: geometry, missing/extra files, label values, file endings. Recommended once per dataset |
| `--skip_fingerprint` | off | Jump straight to planning, reusing the existing `dataset_fingerprint.json` |
| `--only_fingerprint` | off | Extract the fingerprint and exit before planning |
| `--no_pp` | off | Fingerprint and plan only, skip preprocessing |
| `--use_nii` | off | Skip preprocessing entirely: replace each `nnUNetPlans_<config>` folder with symlinks to the raw `imagesTr/*.nii.gz` (the `_0000` channel suffix is dropped). For pipelines that read NIfTI directly instead of `.npy` |

**Cleaning and resuming**

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `--clean_fp` | off | Recompute the fingerprint even if one exists — required after changing the dataset or the fingerprint extractor. Without it, an existing fingerprint is reused |
| `--clean_pf` | off | Delete the `nnUNetPlans_<config>` folders before preprocessing, forcing everything to be re-preprocessed |
| `--clean_split` | off | Delete `splits_final.json` so a fresh cross-validation split is drawn |
| `--continue_` | off | Resume an interrupted fingerprint extraction instead of starting over |
| `--continue_process` | off | Resume preprocessing: cases that already have their `.npz`/`.npy` **and** `.pkl` written are skipped |

**Fingerprint tuning**

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `-num_foreground_voxels_for_intensitystats` | `1e8` | Total number of foreground voxels sampled across the dataset for the intensity statistics. Lower it if fingerprinting runs out of RAM |
| `-keys_percent` | `None` (all cases) | Use only this fraction of the cases for fingerprinting. Mainly for quick tests |
| `--extract_input_image_statistics` | off | Also record per-image input statistics in the fingerprint |

The same work can be split into three separate calls, which is useful when you want to edit the plans
by hand before preprocessing:

```bash
nnUNetv2_extract_fingerprint -d 998 --verify_dataset_integrity
nnUNetv2_plan_experiment     -d 998
nnUNetv2_preprocess          -d 998 -c 3d_fullres -plans_name nnUNetPlans
```

The flag names differ between these three and the all-in-one command: `nnUNetv2_extract_fingerprint`
calls its worker count `-np` (not `-npfp`) and its overwrite switch `--clean` (not `--clean_fp`),
`nnUNetv2_preprocess` calls the plans file `-plans_name` (not `-overwrite_plans_name`), and none of
the three accept the staging, cleaning, resuming or fingerprint-tuning flags listed above — those
exist only on `nnUNetv2_plan_and_preprocess`. Run any of them with `--help` for the authoritative
list.

### 5. What you get

```
nnUNet_preprocessed
└── Dataset998_mm_whs
    ├── dataset.json
    ├── dataset_fingerprint.json
    ├── nnUNetResEncUNetLPlans.json              # patch size, batch size, spacing, architecture per configuration
    ├── splits_final.json                        # the 5-fold cross-validation split
    ├── gt_segmentations/                        # reference labels used for validation
    ├── nnUNetPlans_3d_fullres/
    │     ├── ct_train_1001_.npz                 # data + seg
    │     ├── ct_train_1001_.pkl                 # image properties
    └──   └── ...

```

Cases without a label (unlabeled data, e.g. for self-supervised pretraining) are written as unpacked
`CASE.npy` instead of `CASE.npz`. Both layouts are read transparently at training time.

Only the configurations you preprocessed can be trained. The 2D models in the zoo need `2d`, the 3D
models need `3d_fullres`.

## Training

### Basic command

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET CONFIGURATION FOLD -tr TRAINER
```

For example, training the small U²Net on the 2D configuration, fold 0:

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset998_mm_whs 3d_fullres 0 -tr nnUNetTrainerU2NetMulti
```

- `DATASET` — dataset ID (`998`) or full name (`Dataset998_mm_whs`)
- `CONFIGURATION` — `2d`, `3d_fullres`, `3d_lowres`, `3d_cascade_fullres`
- `FOLD` — `0`–`4` for the 5-fold cross-validation, or `all` to train on every case
- `-tr` — trainer class, i.e. the model. Defaults to `nnUNetTrainer` (the plain nnU-Net baseline)

Pick the GPU with `CUDA_VISIBLE_DEVICES`, not with `-device`; `-device` only chooses between
`cuda`, `cpu` and `mps`.

### Which model? Pick a trainer

Every model in the zoo is a trainer class under
[nnunetv2/training/nnUNetTrainer/](nnunetv2/training/nnUNetTrainer/). They are found by class name
regardless of the file or subfolder they live in, so a custom trainer only has to be placed somewhere
inside that package. 

**CNN**

| Trainer | Model |
|:--------|:------|
| `nnUNetTrainer` | nnU-Net baseline; architecture taken from the plans file |
| `nnUNetTrainerU2NetMulti` / `nnUNetTrainerU2NetMultiSmall` | U²Net / U²Net Small |


**Transformer**

| Trainer | Model |
|:--------|:------|
| `nnUNetTrainerUNETR` | UNETR |
| `nnUNetTrainerSwUNETR` | Swin-UNETR |
| `nnUNetTrainerSwinTransformerUnet` | Swin-Transformer U-Net (SwT) |
| `nnUNetTrainerUNETR2Net` | UNETR²Net |
| `nnUNetTrainerSwT2Net` | SwT²Net |

**Mamba / state-space**

| Trainer | Model |
|:--------|:------|
| `nnUNetTrainerSwinUMamba` / `nnUNetTrainerSwinUMambaScratch` | Swin-UMamba, with / without pretrained encoder |
| `nnUNetTrainerSwinUMambaD` / `nnUNetTrainerSwinUMambaDScratch` | Swin-UMamba† (Mamba-based decoder) |
| `nnUNetTrainerSegMamba` | SegMamba |
| `nnUNetTrainerLightSS2DMambaUNet` | Lightweight SS2D Mamba U-Net |
| `nnUNetTrainerSSND2Net` / `nnUNetTrainerSSND2NetP` | SS2D²Net / SS2D²NetS (cross-scan selective scan) |
| `nnUNetTrainerM2Net` / `nnUNetTrainerM2NetP` | M²Net / M²NetS (SS2D blocks in a U²Net topology) |
| `nnUNetTrainerM2NetModified` / `nnUNetTrainerM2NetPModified` | M²Net variant with a modified block layout |
| `nnUNetTrainerLM2Net` / `nnUNetTrainerLM2NetP` | AltM²Net / AltM²NetS (alternating 1D Mamba scan) |
| `nnUNetTrainerLightMamba2Net` / `nnUNetTrainerLightMamba2NetP` | Mamba2-based light X²Net |
| `nnUNetTrainerPseudoLightMamba2NetP` | Convolutional stand-in for the light Mamba2 model (no `mamba_ssm` needed) |
| `nnUNetTrainerMambaND2Net` / `nnUNetTrainerMambaND2NetP` | MambaND²Net / MambaND²NetS |
| `nnUNetTrainerLightMUNet` | LightM-UNet |

Mamba trainers require a working mamba-ssm installed and a CUDA GPU. Check [mamba installation](./mamba_assets/)

**Data-augmentation variants** (CT metal-artifact, step and motion simulation) live in
[nnunetv2/training/nnUNetTrainer/segmentation/variants/data_augmentation/](nnunetv2/training/nnUNetTrainer/segmentation/variants/data_augmentation/),
e.g. `nnUNetTrainerMetalArtifactStepMotion`, `nnUNetTrainerMetalArtifactStepMotionNoMirrorU2Net`.


### Useful flags

| Flag | Meaning |
|:-----|:--------|
| `-p nnUNetPlans` | Plans identifier; must match the plans file you preprocessed with |
| `-pretrained_weights /path/checkpoint.pth` | Initialize from a checkpoint. Matching keys with matching shapes are transferred, segmentation heads are skipped |
| `-num_gpus 2` | Multi-GPU training via DDP on one node |
| `-device cuda\|cpu\|mps` | Device type |
| `--npz` | Also save softmax outputs of the final validation; needed for ensembling and `nnUNetv2_find_best_configuration` |
| `--c` | Continue an interrupted training from the latest checkpoint |
| `--val` | Skip training and only run the final validation (training must have finished) |
| `--val_best` | Validate with `checkpoint_best.pth` instead of `checkpoint_final.pth` |
| `--use_compressed` | Train directly from the compressed `.npz` files (saves disk, costs CPU and RAM) |
| `--disable_checkpointing` | Do not write checkpoints — handy for quick debug runs |

### Output layout

```
nnUNet_results
└── Dataset998_mm_whs
    └── nnUNetTrainerU2NetMultiSmall__nnUNetResEncUNetLPlans__3d_fullres      # TRAINER__PLANS__CONFIGURATION
        ├── dataset.json
        ├── plans.json
        └── fold_0
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            ├── progress.png                          # loss / pseudo-Dice curves
            ├── training_log_....txt
            └── validation/                            # predictions on the validation split
```

The `TRAINER__PLANS__CONFIGURATION` folder is the model folder you point inference and evaluation at.

To cover the full cross-validation, run all five folds (sequentially, or in parallel on separate
GPUs):

```bash
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset998_mm_whs 2d $fold -tr nnUNetTrainerU2NetMultiSmall --npz
done
```

### Self-supervised pretraining and fine-tuning

The self-supervised (MAE) recipe is organized as two stages:

1. **Pretrain** on unlabeled data with a trainer from
   [nnunetv2/training/nnUNetTrainer/ssl/](nnunetv2/training/nnUNetTrainer/ssl/), e.g.
   `nnUNetTrainerCT3DSSLMAENNUNetP256_BS96`. Set `"target_type": "ssl"` in `dataset.json`; the
   dataset needs no `labelsTr`.
2. **Fine-tune** on the labeled dataset with the matching segmentation trainer from
   [nnunetv2/training/nnUNetTrainer/segmentation/ssl_finetune/](nnunetv2/training/nnUNetTrainer/segmentation/ssl_finetune/),
   passing the pretrained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset064_Cardiac_CT_FM nnUNetResEncUNetLPlans__3d_fullres 3d_fullres 0 \
    -tr nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_100Percent \
    -pretrained_weights /path/to/ssl_checkpoint.pth 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset998_mm_whs nnUNetResEncUNetLPlans__3d_fullres 3d_fullres 0 \
    -tr nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_100Percent \
    -pretrained_weights /path/to/ssl_checkpoint.pth
```

These trainers freeze the encoder for the first epochs, then unfreeze it and continue with a
warmup + poly learning-rate schedule. The `_XXPercent` suffix subsamples the training split
(`01`, `05`, `10`, `20`, `50`, `75`, `100` percent), which is how the low-data experiments are run;
the `_Pretrain_` variants are the ones meant to start from self-supervised weights, the plain
`..._XXPercent` ones train from scratch on the same subsample.



## Citation

Please kindly cite the following paper if you use this repository.

```commandline
@article{kazaj2025claims,
  author    = {Pooya Mohammadi Kazaj and Giovanni Baj and Yazdan Salimi and Anselm W. Stark and Waldo Valenzuela
                 and George C. M. Siontis and Habib Zaidi and Mauricio Reyes and Christoph Graeni and Isaac Shiri},
  title     = {From Claims to Evidence: A Unified Framework and Critical Analysis of CNN vs. Transformer vs. Mamba in Medical Image Segmentation},
  journal   = {arXiv preprint},
  volume    = {2503.01306},
  year      = {2025},
  archivePrefix = {arXiv},
  primaryClass = {eess.IV},
  doi       = {10.48550/arXiv.2503.01306},
  url       = {https://arxiv.org/abs/2503.01306}
}

@article{kazaj2026unified,
  title  = {A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping:
Human-in-the-Loop Data Annotation, Vision Foundation Model Development, Multicenter Evaluation and Clinical Validation},
  author = {Mohammadi Kazaj, Pooya and Weber, Leo Fridolin and Xie, Wen and Safavi-Naini, Seyed Amir Ahmad and Stark,
Anselm and Baj, Giovanni and Mokhtari, Ali and Yoshida, Toshiya and Ryffel, Christoph and Okuno, Taishi and Akashi,
Yoshihiro and Buechel, Ronny R. and Pilgrim, Thomas and Valenzuela, Waldo and Siontis, George C. M. and Xu, Xiaowei
and Hundertmark, Moritz and Windecker, Stephan and Grani, Christoph and Shiri, Isaac},
  journal = {arXiv preprint arXiv:2607.11287},
  year   = {2026},
  doi    = {10.48550/arXiv.2607.11287},
  url    = {https://arxiv.org/abs/2607.11287}
}

@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens
and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z},
  url     = {https://doi.org/10.1038/s41592-020-01008-z}
}
```