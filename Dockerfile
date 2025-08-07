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
# Install PyTorch first to avoid version conflicts
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ------- Clone GraspMolmo source code -------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/samiazirar/GraspMolmo.git
WORKDIR /opt/GraspMolmo

# ------- Install SAM2 separately to handle naming issue -------
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

# ------- Python dependencies (excluding problematic ones) -------
# Install base dependencies first
RUN pip install --no-cache-dir -e .

# ------- Install datagen dependencies with version adjustments -------
RUN pip install --no-cache-dir \
    "acronym_tools @ git+https://github.com/abhaybd/acronym.git" \
    "boto3~=1.36.17" \
    "fastapi~=0.115.7" \
    "h5py~=3.12.1" \
    "openai~=1.64.0" \
    "pydantic~=2.10.5" \
    "pyrender~=0.1.45" \
    "requests~=2.32.3" \
    "scipy~=1.14.1" \
    "trimesh~=4.5.3" \
    "types-boto3~=1.36.17" \
    "types-boto3-s3~=1.36.15" \
    "uvicorn~=0.34.0" \
    "scene-synthesizer~=1.13.1" \
    "hydra-core~=1.3.2" \
    "einops~=0.8.1" \
    "datasets~=3.3.2" \
    "shortuuid~=1.0.13"

# ------- Install open3d (use available version) -------
RUN pip install --no-cache-dir open3d==0.18.0

# ------- Try to install learning3d (may fail due to open3d version) -------
# learning3d requires open3d==0.17.0 which is not available
# We'll skip it for now or install without strict version checking
RUN pip install --no-cache-dir learning3d --no-deps || echo "Warning: learning3d installation failed, continuing without it"

# ------- Install remaining infer dependencies -------
RUN pip install --no-cache-dir \
    transformers~=4.52.4 \
    tensorflow~=2.0 \
    accelerate~=1.7.0

# ------- Clean up build caches -------
RUN apt-get clean && rm -rf /root/.cache/pip

# ------- Default entrypoint -------
CMD ["/bin/bash"]