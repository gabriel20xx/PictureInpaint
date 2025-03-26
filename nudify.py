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
import sys
import io
import time

# Use python 3.11 for this script

# Run this once
# pip install torch diffusers transformers pillow numpy opencv-python accelerate sentencepiece

# ======== CONFIGURATION ========
INPUT_IMAGE_PATH = "example_input_images/input_image_6.jpg"
MASK_OUTPUT_PATH = "example_output_masks/clothes_mask_alpha_6.png"
INPAINTED_OUTPUT_PATH = "example_output_images/nudified_output_6.png"
LOCAL_FLUX_MODEL_PATH = "flux_diffusers_model"  # Path to local model folder
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 3.5  # Default guidance scale (adjustable)
SAMPLER_NAME = "Euler"  # Change this to the desired sampler
MASK_GROW_PIXELS = 15  # Amount to grow (dilate) mask
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2048

# Force UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Global variable to store the pipeline instance
pipe = None

torch.set_grad_enabled(False)  # ✅ Disable gradients globally


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

    with torch.no_grad():
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
def resize_to_fhd_keep_aspect(image):
    global TARGET_WIDTH, TARGET_HEIGHT
    original_width, original_height = image.size

    # Only resize if the original image's width and height are larger than the target dimensions
    if original_width > TARGET_WIDTH and original_height > TARGET_HEIGHT:
        aspect_ratio = original_width / original_height

        if (TARGET_WIDTH / TARGET_HEIGHT) > aspect_ratio:
            new_height = TARGET_HEIGHT
            new_width = int(aspect_ratio * TARGET_HEIGHT)
        else:
            new_width = TARGET_WIDTH
            new_height = int(TARGET_WIDTH / aspect_ratio)

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
    image_path, mask_path, model_path, sampler_name, num_inference_steps, guidance_scale
):
    # Load the converted Flux model
    print("Loading converted Flux model...")

    # Check for available device (GPU if possible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load the model with correct dtype based on the available device
    pipe = FluxFillPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    # Set the sampler (scheduler)
    safe_print(f"🟡 Setting scheduler {sampler_name}...")
    pipe.scheduler = get_scheduler(sampler_name, pipe.scheduler)
    safe_print(pipe.scheduler.config)
    safe_print("✅ Scheduler set")

    # Enable Karras sigmas
    # pipe.scheduler.config["use_karras_sigmas"] = True
    # pipe.scheduler.config["use_exponential_sigmas"] = True
    pipe.scheduler.config["use_beta_sigmas"] = True

    # Optionally, also adjust other parameters if needed
    # pipe.scheduler.config["invert_sigmas"] = False  # Leave as False or adjust as needed

    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    safe_print(f"✅ FluxFillPipeline loaded on {device}.")

    # Load original image & mask
    original_image = Image.open(image_path).convert("RGB")
    original_mask = Image.open(mask_path).convert("L")

    # Resize both to FHD keeping aspect ratio
    image = resize_to_fhd_keep_aspect(original_image)
    mask = resize_to_fhd_keep_aspect(original_mask)

    prompt = "naked body, realistic skin texture, no clothes"

    retries = 0
    while True:
        try:
            safe_print("🟢 Starting inpainting process...")
            # Ensure gradients are disabled during inference
            with torch.inference_mode():  # ✅ Wrap inpainting execution in inference mode
                # Alternatively, you can also use `set_grad_enabled(False)` inside the inference block
                with torch.no_grad():  # Additional safety check to make sure gradients are not tracked
                    result = pipe(
                        prompt=prompt,
                        image=image,
                        mask_image=mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,  # <-- Added guidance scale
                    ).images[0]

            # Generate a unique output path if the file exists
            output_path = get_unique_output_path(INPAINTED_OUTPUT_PATH)

            result.save(output_path)
            safe_print(
                f"✅ Inpainting completed. Output saved as '{INPAINTED_OUTPUT_PATH}'."
            )
            break
        except Exception as e:
            retries += 1
            safe_print(f"❌ Inpainting failed (attempt {retries}). Error: {e}")
            time.sleep(5)


# ======== MAIN FUNCTION ========
def main():
    try:
        image = load_image(INPUT_IMAGE_PATH)
        processor, model = load_segmentation_model()
        clothes_mask = generate_clothing_mask(model, processor, image)
        save_black_inverted_alpha(clothes_mask, MASK_OUTPUT_PATH, MASK_GROW_PIXELS)

        # Retry inpainting process indefinitely with guidance scale
        inpaint(
            INPUT_IMAGE_PATH,
            MASK_OUTPUT_PATH,
            LOCAL_FLUX_MODEL_PATH,
            SAMPLER_NAME,
            NUM_INFERENCE_STEPS,
            GUIDANCE_SCALE,
        )
    except Exception as e:
        safe_print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
