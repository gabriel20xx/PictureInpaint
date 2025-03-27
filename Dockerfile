# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio \
    diffusers transformers pillow numpy opencv-python accelerate sentencepiece peft xformers \
    gradio bitandbytes

# Expose Gradio port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "nudify.py"]
