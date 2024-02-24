# Installation

We suggest you use an Ubuntu 18.04 / 20.04 OS and CUDA 11.3 system. Installation may vary based on CUDA and OS.

There are two methods to install CircleSnake:
- From source
- From Docker

## Source
### Set up the Python environment
Install [conda](https://docs.conda.io/en/latest/miniconda.html)
```
conda env create -f environment.yml
conda activate CircleSnake
```
### Cuda Toolkit
Install [cudatoolkit](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu)

```
# Add Cuda toolkit to Path in ```.bashrc```
# Export CUDA_HOME based on your CUDA version
# example: export CUDA_HOME="/usr/local/cuda-11.3"
```

### Compile cuda extensions under `lib/csrc`

```
export ROOT=/path/to/snake
cd $ROOT/lib/csrc

cd DCNv2_latest/
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```

## Docker
```
docker run -it --gpus all bluenotebook/circlesnake
```
