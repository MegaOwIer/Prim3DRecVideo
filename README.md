# Prim3DRecVideo

## Description

Team project for CS7352 of SJTU.

## Environment Setup

```sh
# Install the packages listed in `environment.yml`
conda env create -f environment.yml

conda activate prim3d

# Install thirdparty libraries
git submodule init
git submodule update
cd thirdparty
python -m pip install -e detectron2/
```

## Usage

[Explain how to use your project, including any necessary commands or configurations.]

## License

[Specify the license under which your project is released.]

## Contact

[Provide contact information for users to reach out to you.]
