# Prim3DRecVideo

## Description

Team project for CS7352 of SJTU.

## Environment Setup

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
pip install -e nvdiffrast/
```

## Usage

To run this project, execute the following command in conda environment

```sh
python main.py [...args]
```

You may need to extract the `D3DHOI` dataset into `datasets/d3dhoi_video_data/` to make sure the default values of arguments work.

After training and testing, you will get the predicted position info in `test/baseline/output_dir/laptop/visualize_results`, and you can use `ply2mp4.py` to visualize it.
