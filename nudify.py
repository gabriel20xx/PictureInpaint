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

# Run this once
# pip install torch diffusers transformers pillow numpy opencv-python accelerate sentencepiece

# Force UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ======== CONFIGURATION ========
INPUT_IMAGE_PATH = "example_input_images/input_image_4.jpg"
MASK_OUTPUT_PATH = "example_output_masks/clothes_mask_alpha_4.png"
INPAINTED_OUTPUT_PATH = "example_output_pictures/nudified_output_4.png"
MASK_GROW_PIXELS = 10  # Amount to grow (dilate) mask


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
        safe_print("✅ Segmentation model loaded.")
        return processor, model
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load segmentation model: {e}")


# ======== GENERATE CLOTHES MASK ========
def generate_clothing_mask(model, inputs, original_size):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    segmentation = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Clothing classes: 1,4,5,6,7
    clothing_indices = {1, 4, 5, 6, 7}
    clothes_mask = np.isin(segmentation, list(clothing_indices)).astype(np.uint8)

    # Resize to original image size
    clothes_mask_resized = cv2.resize(
        clothes_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST,
    )
    return clothes_mask_resized


# ======== APPLY MASK GROW AND SAVE ========
def save_black_inverted_alpha(clothes_mask, output_path):
    h, w = clothes_mask.shape
    black = Image.new("RGB", (w, h), (0, 0, 0))

    # Invert mask: Clothes = 0 (transparent), Rest = 255 (opaque)
    alpha = np.where(clothes_mask == 1, 0, 255).astype(np.uint8)

    # --- Grow the mask using dilation ---
    kernel = np.ones((MASK_GROW_PIXELS, MASK_GROW_PIXELS), np.uint8)
    alpha = cv2.dilate(alpha, kernel, iterations=1)

    # Save
    alpha_image = Image.fromarray(alpha)
    result = black.convert("RGBA")
    result.putalpha(alpha_image)
    result.save(output_path)
    safe_print(f"✅ Mask (grown, no blur) saved: {output_path}")


def resize_to_fhd_keep_aspect(image, target_width=1920, target_height=1080):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Compute new size maintaining aspect ratio
    if (target_width / target_height) > aspect_ratio:
        # Fit to height
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        # Fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


# ======== INPAINTING FUNCTION ========
def inpaint_with_retry(image_path, mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_print(f"🟢 Using device: {device}")

    retries = 0
    while True:
        try:
            safe_print("🟡 Loading FluxFillPipeline...")
            pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev")
            pipe = pipe.to(device)
            safe_print("✅ FluxFillPipeline loaded.")
            break
        except Exception as e:
            retries += 1
            safe_print(f"❌ Failed to load pipeline (attempt {retries}). Error: {e}")
            time.sleep(5)

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
            safe_print("🟢 Starting inpainting...")
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=30,
            ).images[0]

            # Ensure same size output
            result = result.resize(image.size)

            result.save(INPAINTED_OUTPUT_PATH)
            safe_print(f"✅ Inpainting done. Saved as '{INPAINTED_OUTPUT_PATH}'.")
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
        inputs = processor(images=image, return_tensors="pt")
        clothes_mask = generate_clothing_mask(model, inputs, image.size)

        save_black_inverted_alpha(clothes_mask, MASK_OUTPUT_PATH)
        inpaint_with_retry(INPUT_IMAGE_PATH, MASK_OUTPUT_PATH)
    except Exception as e:
        safe_print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
