#FROM --platform=linux/amd64 pytorch/pytorch
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS="ignore"


RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  wget \
  unzip \
  libopenblas-dev \
  python3.10 \
  python3.10-dev \
  python3-pip \
  nano \
  && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/* 

# Update the python3 and pip3 symlinks to point to Python 3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
# RUN ln -sf /usr/bin/pip3 /usr/bin/pip3

# Verify the Python version
RUN python3 -V  # This should now output Python 3.10.x
RUN pip3 -V     # This should now use pip for Python 3.10

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"

USER user

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python3 -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

### Copy and install custom packages in editable mode
COPY --chown=user:user ./src/dynamic-network-architectures_global_rad/dynamic-network-architectures/ /opt/algorithm/dynamic-network-architectures/
COPY --chown=user:user ./src/nnunetv2_global_rad/ /opt/algorithm/nnunetv2_global_rad/
COPY --chown=user:user ./src/pyradiomics-3.1.0-Zengtian/ /opt/algorithm/pyradiomics-3.1.0-Zengtian/
COPY --chown=user:user ./src/pytorchradiomics-main/ /opt/algorithm/pytorchradiomics-main/
COPY --chown=user:user ./src/report-guided-annotation/ /opt/algorithm/report-guided-annotation/


# Install packages in editable mode
RUN python3 -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    -e /opt/algorithm/dynamic-network-architectures/ \
    -e /opt/algorithm/nnunetv2_global_rad/ \
    -e /opt/algorithm/pyradiomics-3.1.0-Zengtian/ \
    -e /opt/algorithm/pytorchradiomics-main/ \
    -e /opt/algorithm/report-guided-annotation/ && \
    rm -rf ~/.cache/pip

# Install a few dependencies that are not automatically installed
RUN python3 -V
RUN python -V
RUN pip -V
RUN pip3 -V
RUN pip3 install \
        graphviz \
        onnx \
        acvl_utils==0.2 \
        SimpleITK && \
    rm -rf ~/.cache/pip

COPY --chown=user:user ./src/nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

### Define workdir
WORKDIR /opt/app

COPY --chown=user:user ./src/process.py /opt/app/
COPY --chown=user:user ./src/data_utils.py /opt/app/
COPY --chown=user:user ./src/__init__.py /opt/app/
COPY --chown=user:user ./src/PANORAMA_voxel.json /opt/app/
COPY --chown=user:user ./src/PANORAMA_0307.yaml /opt/app/
COPY --chown=user:user ./src/voxel_radiomics.py /opt/app/

### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT [ "python3.10", "-m", "process" ]
