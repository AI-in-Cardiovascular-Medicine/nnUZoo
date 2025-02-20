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