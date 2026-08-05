# Mamba environment (SwinUMamba)

SwinUMamba is the one architecture that cannot be trained in the main
environment. This folder holds its environment definition; 

## Why a separate environment

The Mamba kernels (`mamba-ssm`, `causal-conv1d`) compile CUDA extensions against a fixed
torch version and we built it against **torch 2.4**, whereas the remaining models were trained
with a current torch. Rather than holding the whole benchmark back at torch 2.4, SwinUMamba
gets its own environment and everything else runs in the main one.


## Installation

The CUDA extensions must be compiled against a torch that is *already installed*, so this
is necessarily a two-step install — the second step runs with `--no-build-isolation`
precisely so that pip does not pull its own torch into an isolated build env.

```bash
conda create -n mamba python=3.12 -y
conda activate mamba


git clone https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo.git
cd nnUZoo
pip install .

cd mamba_assets
pip uninstall torch torchvision transformers
pip install .
pip install torch==2.4 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57
# Download 
# https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# and 
# https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### Note
- The build is slow (minutes) and needs a few GB of RAM per compile job.
- The rest of the preprocessing and training is like the other trainers, You can try `nnUNetTrainerMetalArtifactStepMotionNoMirrorSwinUMamba` and other mamba trainers with the same pipelines.
