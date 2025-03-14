# Use NVIDIA CUDA base image with Python 3.9
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-opencv \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libx264-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Update pip
RUN python -m pip install --upgrade pip

# Install mim
RUN pip install -U openmim

# Create and set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install mmcv
RUN mim install mmcv-full==1.6.2 mmsegmentation==0.27.0

# Install source dependencies
COPY requirements-source.txt ./
RUN pip install -r requirements-source.txt

# Copy Python files
COPY segmentor.py serve.py ./

# Default command
CMD ["python", "serve.py"] 
