# X2Net

<img src="assets/main_img.jpg">


## Contents
- [Overview](#overview-)
- [Project Online Page & Test](#project-online-page--test)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Preprocessing](#preprocessing)
- [Train](#train)
- [Evaluate model](#evaluate-model)
- [Inference](#inference)
- [Inference On Samples](#inference-on-samples-data)
- [Explainability](#explainability)
- [Result Plots](#result-plots)
- [DockerFiles](#dockerfiles)
- [License](#license)
- [Citation](#citation)


# Overview
...

# System Requirements
## HardWare Requirements
- Train Requirements
  - Minimum RAM of 24 GB
  - Minimum CPU of 8 Cores
  - A decent GPU with minimum of 24 GB VRAM such as Nvidia-4090, A6000, A100, etc.
- Inference Arguments
  - Minimum RAM of 8 GB
  - Minimum CPU of 4 Cores
  - For a faster inference and for Mamba models a GPU can be used however it's not necessary.
## Software Requirements
### OS Requirements

The developmental version of the code has been tested on the following systems:
* Linux: Ubuntu 22.04, Ubuntu 24.04, Pop!_OS 22.04
* Mac OSX: Not tested
* Windows: Not tested

The codes with CUDA should be compatible with Windows, Mac, and other Linux distributions.

# Installation Guide

## Cuda Installation
- To install cuda, please refer to [cuda-installation-documents](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- To install Mamba, please refer to [mamba-installation-documents](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation)

## Library Installation
```commandline
pip install -r requirements.txt
pip install ./nnUNet
```

## Train
To train any of the models, the dataset should be in the nnunetv2 format. 
```commandline
├── nnunet_raw
│   ├── Dataset030_AbdomenMR
│   │   ├── imagesTr
│   │   │   ├── amos_0507_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── amos_0507.nii.gz
│   │   │   ├── ...
│   │   ├── imagesTs
│   │   │   ├── amos_0507_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTs
│   │   │   ├── amos_0507.nii.gz
│   │   │   ├── ...
```

Then run the following code to train a model:
```commandline
python train.py --device 1 --dataset_name  Dataset030_AbdomenMR --tr nnUNetTrainerM2NetP --model 2d --num_epochs 250
```
For more variable please run `python train.py --help`

## Test
For test execute the following command:
```commandline

```