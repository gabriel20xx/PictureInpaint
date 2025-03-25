import torch
from diffusers import FluxFillPipeline
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
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

# Force UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ======== CONFIGURATION ========
INPUT_IMAGE_PATH = "example_input_images/input_image_6.jpg"
MASK_OUTPUT_PATH = "example_output_masks/clothes_mask_alpha_6.png"
INPAINTED_OUTPUT_PATH = "example_output_images/nudified_output_6.png"
MASK_GROW_PIXELS = 15  # Amount to grow (dilate) mask


# ======== SAFE PRINT FUNCTION ========
def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode())


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
def resize_to_fhd_keep_aspect(image, target_width=1920, target_height=1080):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if (target_width / target_height) > aspect_ratio:
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)


# ======== INPAINTING FUNCTION ========
def inpaint_with_retry(image_path, mask_path, max_retries=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_print(f"🟢 Using device: {device}")

    for attempt in range(1, max_retries + 1):
        try:
            safe_print(f"🟡 Loading FluxFillPipeline (Attempt {attempt})...")
            pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev"
            ).to(device)
            safe_print("✅ FluxFillPipeline loaded.")
            break
        except Exception as e:
            safe_print(f"❌ Failed to load pipeline (Attempt {attempt}). Error: {e}")
            if attempt == max_retries:
                raise RuntimeError("🚨 Pipeline failed after multiple attempts.")
            time.sleep(5)

    # Load original image & mask
    original_image = Image.open(image_path).convert("RGB")
    original_mask = Image.open(mask_path).convert("L")

    # Resize both to FHD keeping aspect ratio
    image = resize_to_fhd_keep_aspect(original_image)
    mask = resize_to_fhd_keep_aspect(original_mask)

    prompt = "naked body, realistic skin texture, no clothes"

    for attempt in range(1, max_retries + 1):
        try:
            safe_print(f"🟢 Starting inpainting (Attempt {attempt})...")
            result = pipe(
                prompt=prompt, image=image, mask_image=mask, num_inference_steps=30
            ).images[0]
            result = result.resize(original_image.size)  # Ensure same size output
            result.save(INPAINTED_OUTPUT_PATH)
            safe_print(f"✅ Inpainting done. Saved as '{INPAINTED_OUTPUT_PATH}'.")
            return
        except Exception as e:
            safe_print(f"❌ Inpainting failed (Attempt {attempt}). Error: {e}")
            if attempt == max_retries:
                raise RuntimeError("🚨 Inpainting failed after multiple attempts.")
            time.sleep(5)


# ======== MAIN FUNCTION ========
def main():
    try:
        image = load_image(INPUT_IMAGE_PATH)
        processor, model = load_segmentation_model()
        clothes_mask = generate_clothing_mask(model, processor, image)
        save_black_inverted_alpha(clothes_mask, MASK_OUTPUT_PATH, MASK_GROW_PIXELS)
        inpaint_with_retry(INPUT_IMAGE_PATH, MASK_OUTPUT_PATH)
    except Exception as e:
        safe_print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
