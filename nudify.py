import torch
from diffusers import (
    FluxFillPipeline,
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import (
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
)
from PIL import Image
import os
import numpy as np
import cv2
import gc
import sys
import io
import time

# Use python 3.11 for this script

# Run these once
# pip install torch --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers pillow numpy opencv-python accelerate sentencepiece peft xformers gc

# ======== CONFIGURATION ========
INPUT_IMAGE_PATH = "example_input_images/input_image_6.jpg"
MASK_OUTPUT_PATH = "example_output_masks/clothes_mask_alpha_6.png"
INPAINTED_OUTPUT_PATH = "example_output_images/nudified_output_6.png"
USE_LOCAL_MODEL = False
LOCAL_FLUX_MODEL_PATH = "models/converted_model"  # Path to local model folder
REMOTE_FLUX_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
USE_LORA = True
REMOTE_LORA = "CultriX/flux-nsfw-highress"
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5  # Default guidance scale (adjustable)
SAMPLER_NAME = "Euler"  # Change this to the desired sampler
MASK_GROW_PIXELS = 15  # Amount to grow (dilate) mask
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2048

# Force UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ======== SAFE PRINT FUNCTION ========
def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode())


# Function to avoid overwriting existing files by adding a suffix
def get_unique_output_path(base_path):
    if not os.path.exists(base_path):
        return base_path

    # Add a numerical suffix to the file if it already exists
    name, ext = os.path.splitext(base_path)
    counter = 1
    new_path = f"{name}_{counter}{ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{name}_{counter}{ext}"

    return new_path


# ======== LOAD INPUT IMAGE ========
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file '{image_path}' not found.")
    try:
        image = Image.open(image_path).convert("RGB")
        safe_print(f"✅ Loaded image: {image_path}")
        return image
    except Exception as e:
        raise IOError(f"❌ Failed to load image: {e}")


# ======== LOAD SEGMENTATION MODEL ========
def load_segmentation_model():
    try:
        safe_print("🟡 Loading clothes segmentation model...")
        processor = SegformerImageProcessor.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        )
        model.eval()
        safe_print("✅ Segmentation model loaded.")
        return processor, model
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load segmentation model: {e}")


# ======== GENERATE CLOTHES MASK ========
def generate_clothing_mask(model, processor, image):
    device = torch.device("cpu")  # Force CPU inference
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    logits = outputs.logits
    segmentation = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Clothing classes: 1,4,5,6,7
    clothing_indices = {1, 4, 5, 6, 7}
    clothes_mask = np.isin(segmentation, list(clothing_indices)).astype(np.uint8)

    # Resize to original image size
    clothes_mask_resized = cv2.resize(
        clothes_mask, image.size, interpolation=cv2.INTER_NEAREST
    )

    if np.count_nonzero(clothes_mask_resized) == 0:
        raise ValueError("⚠️ No clothing detected in the image.")

    return clothes_mask_resized


# ======== APPLY MASK GROW AND SAVE ========
def save_black_inverted_alpha(clothes_mask, output_path, mask_grow_pixels=15):
    mask = (clothes_mask * 255).astype(np.uint8)  # Convert 1s to 255 (white mask)

    dilate_size = max(5, mask_grow_pixels)  # Ensure mask grows sufficiently
    close_size = max(3, mask_grow_pixels // 3)  # Ensure minimum size of 3

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )

    # --- Expand the mask slightly to remove unwanted artifacts ---
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # --- Smooth edges to avoid harsh transitions ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Apply a Gaussian blur for extra smoothing
    blur_ksize = max(5, (mask_grow_pixels // 2) * 2 + 1)  # Ensure odd kernel size
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), sigmaX=0, sigmaY=0)

    # Save as grayscale PNG
    Image.fromarray(mask).save(output_path)
    safe_print(f"✅ Fixed mask saved as grayscale: {output_path}")


# ======== RESIZE FUNCTION FOR INPAINTING ========
def resize_to_fhd_keep_aspect(image, target_width, target_height):
    original_width, original_height = image.size

    # Only resize if the original image's width and height are larger than the target dimensions
    if original_width > target_width and original_height > target_height:
        aspect_ratio = original_width / original_height

        if (target_width / target_height) > aspect_ratio:
            new_height = target_height
            new_width = int(aspect_ratio * target_height)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        return image.resize((new_width, new_height), Image.LANCZOS)
    else:
        # If the original image is smaller, return the original image without resizing
        return image


# ======== FUNCTION TO SELECT SAMPLER ========
def get_scheduler(scheduler_name, default_scheduler):
    """Returns the correct sampler based on the user's choice"""

    schedulers = {
        "Euler": FlowMatchEulerDiscreteScheduler.from_config(default_scheduler.config),
        "DPM++ 2M": DPMSolverMultistepScheduler.from_config(default_scheduler.config),
        # Add more samplers here if needed
    }

    new_scheduler = schedulers.get(scheduler_name, default_scheduler)

    # Ensure backward compatibility with older versions
    if hasattr(new_scheduler, "set_timesteps"):
        scheduler_params = new_scheduler.set_timesteps.__code__.co_varnames
        if "mu" not in scheduler_params:
            new_scheduler.set_timesteps = (
                lambda *args, **kwargs: default_scheduler.set_timesteps(
                    *args, **{k: v for k, v in kwargs.items() if k != "mu"}
                )
            )

    return new_scheduler


# ======== INPAINTING FUNCTION ========
def inpaint(
    image_path,
    mask_path,
    use_local_model,
    model_path,
    remote_model,
    sampler_name,
    num_inference_steps,
    guidance_scale,
    target_width,
    target_height,
    inpainted_output_path,
    use_lora,
    remote_lora,
):
    # Load the Flux model
    safe_print("Loading Flux model...")

    # Check for available device (GPU if possible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load the model with correct dtype based on the available device
    if use_local_model:
        pipe = FluxFillPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        safe_print(f"✅ Checkpoint {model_path} applied")
    else:
        pipe = FluxFillPipeline.from_pretrained(remote_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        safe_print(f"✅ Checkpoint {remote_model} applied.")

    if use_lora:
        pipe.load_lora_weights(remote_lora)
        safe_print(f"✅ Lora {remote_lora} applied.")

    # Set the sampler (scheduler)
    safe_print(f"🟡 Setting scheduler {sampler_name}...")
    pipe.scheduler = get_scheduler(sampler_name, pipe.scheduler)
    safe_print("✅ Scheduler set")

    # Enable Karras sigmas
    # pipe.scheduler.config["use_karras_sigmas"] = True
    # pipe.scheduler.config["use_exponential_sigmas"] = True
    pipe.scheduler.config["use_beta_sigmas"] = True

    # Optionally, also adjust other parameters if needed
    # pipe.scheduler.config["invert_sigmas"] = False  # Leave as False or adjust as needed

    safe_print(f"Config: {pipe.scheduler.config}")

    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Free GPU memory
    gc.collect()  # Free CPU memory

    # Force CPU if needed
    if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 6 * 1024 * 1024 * 1024:
        safe_print("⚠️ Low VRAM detected, switching to CPU mode.")
        pipe.to("cpu")

    # Ensure model is on the correct device
    pipe.to(device)

    safe_print(f"✅ FluxFillPipeline loaded on {device}.")

    # Load original image & mask
    original_image = Image.open(image_path).convert("RGB")
    original_mask = Image.open(mask_path).convert("L")

    # Resize both to FHD keeping aspect ratio
    image = resize_to_fhd_keep_aspect(original_image, target_width, target_height)
    mask = resize_to_fhd_keep_aspect(original_mask, target_width, target_height)

    prompt = "naked body, realistic skin texture, no clothes, nude, no bra, no top, no panties, no pants, no shorts"

    try:
        safe_print("🟢 Starting inpainting process...")
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Generate a unique output path if the file exists
        output_path = get_unique_output_path(inpainted_output_path)

        # Ensure same size output
        result = result.resize(image.size)

        result.save(output_path)
        safe_print(f"✅ Inpainting completed. Output saved as '{output_path}'.")
    except Exception as e:
        safe_print(f"❌ Inpainting failed. Error: {e}")
        time.sleep(5)


# ======== MAIN FUNCTION ========
def main():
    try:
        print(torch.cuda.is_available())  # Should return True if CUDA is working
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))  # Should print GTX 1080
        
        image = load_image(INPUT_IMAGE_PATH)
        processor, model = load_segmentation_model()
        clothes_mask = generate_clothing_mask(model, processor, image)
        save_black_inverted_alpha(clothes_mask, MASK_OUTPUT_PATH, MASK_GROW_PIXELS)

        # Retry inpainting process indefinitely with guidance scale
        inpaint(
            INPUT_IMAGE_PATH,
            MASK_OUTPUT_PATH,
            USE_LOCAL_MODEL,
            LOCAL_FLUX_MODEL_PATH,
            REMOTE_FLUX_MODEL,
            SAMPLER_NAME,
            NUM_INFERENCE_STEPS,
            GUIDANCE_SCALE,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            INPAINTED_OUTPUT_PATH,
            USE_LORA,
            REMOTE_LORA,
        )
    except Exception as e:
        safe_print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
