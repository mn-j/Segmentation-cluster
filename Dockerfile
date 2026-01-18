
ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8

# Python build version (matches your maintainer template pattern)
ARG PYTHON_VERSION=3.11.3

# Pick a PyTorch version that exists for cu118.
# If you want max portability, align with maintainer default (2.7.1).
# You can override at build time: --build-arg PYTORCH_VERSION=2.7.1
ARG PYTORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1

ARG BUILD_JOBS=16

# =============================================================
# Base image
# =============================================================

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG CUDA_MAJOR_VERSION
ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG BUILD_JOBS

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================
# System dependencies (mostly from maintainer template)
# - Kept OpenCV-friendly libs (libgl1, libglib2.0-0) for opencv-python
# =============================================================

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        man \
        openssh-server \
        ca-certificates apt-transport-https \
        sudo \
        git subversion \
        nano vim \
        tmux screen \
        htop \
        g++ meson ninja-build \
        rsync \
        pv \
        curl wget \
        bzip2 zip unzip \
        dcmtk libboost-all-dev \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx libglib2.0-0 libgirepository1.0-dev libexif-dev librsvg2-dev fftw3-dev orc-0.4-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    cat /etc/sudoers | grep -v secure_path > /tmp/sudoers && mv /tmp/sudoers /etc/sudoers

RUN env | grep '^PATH=\|^LD_LIBRARY_PATH=\|^LANG=\|^LC_ALL=\|^CUDA_ROOT=' > /etc/environment

# Timezone (as in maintainer template)
RUN echo "Europe/Amsterdam" > /etc/timezone && \
    rm -f /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# User (as in maintainer template)
RUN useradd -ms /bin/bash user && \
    (echo user ; echo user) | passwd user && \
    gpasswd -a user _ssh && \
    gpasswd -a user sudo

# =============================================================
# Install Python from source (as in maintainer template)
# =============================================================

RUN cd /tmp && \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd / && \
    rm -rf /tmp/Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip pip-tools wheel setuptools && \
    printf '#!/bin/bash\necho "Please use pip3 instead of pip to install packages for python3"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip && \
    rm -rf ~/.cache/pip

# =============================================================
# Python deps via pip-tools (compatible with maintainer approach)
# =============================================================

# requirements.in should NOT pin torch==...+cuXYZ.
# We'll select the correct wheel set via --index-url using CUDA_MAJOR_VERSION.
COPY requirements.in /root/python-packages/requirements.in

RUN cd /root/python-packages && \
    CUDA_IDENTIFIER_PYTORCH=$(echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5) && \
    pip-compile requirements.in \
      --verbose \
      --index-url "https://download.pytorch.org/whl/${CUDA_IDENTIFIER_PYTORCH}" \
      --extra-index-url "https://pypi.org/simple" && \
    pip-sync && \
    rm -rf ~/.cache/pip*

# TensorFlow env vars kept from template (harmless even if unused)
ENV FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
ENV TF_CPP_MIN_LOG_LEVEL=3
RUN env | grep '^FOR_DISABLE_CONSOLE_CTRL_HANDLER=\|^TF_CPP_MIN_LOG_LEVEL=' >> /etc/environment

STOPSIGNAL SIGINT
EXPOSE 22 6006 8888

# =============================================================
# Copy your training code
# =============================================================

WORKDIR /workspace
COPY . /workspace

# Optional sanity check (fast)
RUN python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# Use maintainer entrypoint style if they require it (ssh/jupyter).
# Otherwise, for pure training images, you can just use CMD.
# Here we keep it simple for training:
USER user
CMD ["python3", "-u", "train_swin.py"]
