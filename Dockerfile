# Use an official NVIDIA CUDA base image with Ubuntu as the operating system
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variable for non-interactive installation
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

# Ensure `python` command is available
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/gabriel20xx/PictureInpaint.git .

RUN pip install -r requirements.txt
# Expose Gradio port
EXPOSE 7860

# Run the Gradio app
CMD ["/usr/bin/python3.11", "nudify.py"]
