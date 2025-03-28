# Use an official NVIDIA CUDA base image with Ubuntu as the operating system
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set the environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
RUN pip install --no-cache-dir \
    diffusers transformers pillow numpy opencv-python accelerate sentencepiece peft xformers \
    gradio

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Expose Gradio port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "nudify.py"]
