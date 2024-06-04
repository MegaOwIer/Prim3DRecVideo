# Prim3DRecVideo

## Description

Team project for CS7352 of SJTU.

## Environment Setup

**Alert**: the `environment.yml` may be NOT up to date, so this setup instruction may not work.

We are using CUDA 11.8 in our test. If you're using a different version of CUDA, you may need to manually adjust version of some packages such as `torch`.

```sh
# Install the packages listed in `environment.yml`
conda env create -f environment.yml

conda activate prim3D

# Install thirdparty libraries
git submodule init
git submodule update
cd thirdparty
pip install -e detectron2/
pip install -e pytorch3d/
```

## Usage

[Explain how to use your project, including any necessary commands or configurations.]
