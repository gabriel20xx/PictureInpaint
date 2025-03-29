# Use an official NVIDIA CUDA base image with Ubuntu as the operating system
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install dependencies
RUN apt update && apt install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip nvidia-cuda-toolkit git \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/gabriel20xx/PictureInpaint.git .

# Make startup script executable
RUN chmod +x startup.sh

# Expose Gradio port
EXPOSE 7860

# Run the Gradio app @
CMD ["./startup.sh"]
