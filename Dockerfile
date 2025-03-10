# Use NVIDIA CUDA base image with Python 3.9
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
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

# Test CUDA
COPY cuda.py ./
RUN python cuda.py

# Install mmcv
RUN mim install mmcv-full==1.7.2

# Install source dependencies
COPY requirements-source.txt ./
RUN pip install -r requirements-source.txt

# Install FastAPI
RUN pip install fastapi python-multipart uvicorn

# Copy Python files
COPY main.py depther.py serve.py ./

# Default command
CMD ["python", "serve.py"] 
