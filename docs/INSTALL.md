## Installation

We used an Ubuntu 18.04 OS and CUDA 11.4 system. Installation may vary based on CUDA and OS.

## Set up the Python environment
0. Install [mamba](https://mamba.readthedocs.io/en/latest/installation.html) (faster) or [conda](https://docs.conda.io/en/latest/miniconda.html)
```
conda env create -f environment.yml
conda activate CircleSnake
```

## Install Apex
```
# Export CUDA_HOME based on your CUDA version
# example: export CUDA_HOME="/usr/local/cuda-11.4"

git clone https://github.com/EthanHNguyen/circlesnake-apex
cd circlesnake-apex
python setup.py install --cuda_ext --cpp_ext
```

## Compile cuda extensions under `lib/csrc`

```
export ROOT=/path/to/snake
cd $ROOT/lib/csrc

# Export CUDA_HOME based on your CUDA version
# example: export CUDA_HOME="/usr/local/cuda-11.4"

cd DCNv2_latest/
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```