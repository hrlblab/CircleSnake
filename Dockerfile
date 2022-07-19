ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.07-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE

RUN git clone https://github.com/hrlblab/CircleSnake.git
WORKDIR CircleSnake

# Create the environment:
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "CircleSnake", "/bin/bash", "-c"]

## Make RUN commands use the new environment:
#RUN echo "conda activate CircleSnake" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]

RUN cd lib/csrc/dcn_v2/ && \
    python setup.py build_ext --inplace && \
    cd ../extreme_utils && \
    python setup.py build_ext --inplace && \
    cd ../roi_align_layer && \
    python setup.py build_ext --inplace
