# syntax=docker/dockerfile:1

# -------------------------------------------------------
# GraspMolmo development image
# Compatible with CUDA 12.2 runtimes (e.g. RTX-40xx on driver ≥ 535.86)
# -------------------------------------------------------

# ------- Base image (CUDA 12.2 + cuDNN 9) -------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

# ------- Common environment flags -------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# ------- System packages -------
RUN apt-get update && \
    # Python 3.11 and build essentials
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        python3-pip python3-venv \
        build-essential git cmake ninja-build \
        # Libraries required by Open3D / OpenCV / PyBullet
        libgl1 libgl1-mesa-glx libglu1-mesa libglvnd0 \
        libxi6 libxrender1 libxext6 libsm6 libglib2.0-0 \
        # Misc utilities
        ca-certificates curl wget ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Upgrade pip/setuptools for the freshly installed Python
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ------- CUDA-matched PyTorch -------
# If you do not need GPU support, comment these two lines and
# install the CPU wheels instead.
RUN pip install --no-cache-dir \
    torch==2.3.0+cu122 \
    torchvision==0.18.0+cu122 \
    --extra-index-url https://download.pytorch.org/whl/cu122

# ------- Clone GraspMolmo source code -------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/samiazirar/GraspMolmo.git
WORKDIR /opt/GraspMolmo

# ------- Python dependencies (all extras) -------
RUN pip install --no-cache-dir -e .[all]

# ------- Clean up build caches -------
RUN apt-get clean && rm -rf /root/.cache/pip

# ------- Default entrypoint -------
CMD ["/bin/bash"]