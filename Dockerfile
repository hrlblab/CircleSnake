ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.02-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE

RUN git clone https://github.com/hrlblab/CircleSnake.git
WORKDIR CircleSnake



RUN conda install mamba -n base -c conda-forge && \
    mamba env create -f environment.yml

RUN conda init bash
SHELL ["conda", "activate", "CircleSnake", "/bin/bash", "-c"]

RUN git clone https://github.com/EthanHNguyen/circlesnake-apex && \
    cd circlesnake-apex && \
    python setup.py install --cuda_ext --cpp_ext && \
    cd ../lib/csrc && \
    cd DCNv2_latest/ && \
    python setup.py build_ext --inplace && \
    cd ../extreme_utils && \
    python setup.py build_ext --inplace && \
    cd ../roi_align_layer && \
    python setup.py build_ext --inplace
