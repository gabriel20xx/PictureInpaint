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

# Force UTF-8 encoding for Windows console to handle Unicode (emojis, etc.)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ======== CONFIGURATION ========
INPUT_IMAGE_PATH = "input_image_4.jpg"
MASK_OUTPUT_PATH = "clothes_mask_alpha_4.png"
INPAINTED_OUTPUT_PATH = "nudified_output_4.png"


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

    # Resize back to original image size (width, height)
    clothes_mask_resized = cv2.resize(
        clothes_mask,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST,
    )
    return clothes_mask_resized


# ======== APPLY SMOOTHING AND SAVE MASK ========
def save_black_inverted_alpha(clothes_mask, output_path):
    h, w = clothes_mask.shape
    black = Image.new("RGB", (w, h), (0, 0, 0))

    # Invert mask: Clothes = 0 (transparent), Rest = 255 (opaque)
    alpha = np.where(clothes_mask == 1, 0, 255).astype(np.uint8)

    # --- Smoothing ---
    # 1. Morphological closing to remove small holes & jagged edges
    kernel = np.ones((5, 5), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

    # 2. Apply Gaussian Blur to smooth out edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), sigmaX=0, sigmaY=0)

    # Convert and save
    alpha_image = Image.fromarray(alpha)
    result = black.convert("RGBA")
    result.putalpha(alpha_image)
    result.save(output_path)
    safe_print(f"✅ Mask with transparent clothes saved: {output_path}")


# ======== INPAINTING FUNCTION WITH INFINITE RETRY ========
def inpaint_with_retry(image_path, mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_print(f"🟢 Using device: {device}")

    retries = 0
    while True:  # Infinite retry loop for loading the pipeline
        try:
            # Load the FLUX.1 Fill model with retries
            safe_print("🟡 Attempting to load the FluxFillPipeline...")
            pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev")
            pipe = pipe.to(device)
            safe_print("✅ FluxFillPipeline loaded.")
            break  # Exit loop once model is successfully loaded
        except Exception as e:
            retries += 1
            safe_print(f"❌ Failed to load pipeline (attempt {retries}). Error: {e}")
            time.sleep(5)  # Wait before retrying

    # Load and preprocess images
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Ensure mask is in grayscale

    # Define your prompt
    prompt = "naked body, realistic skin texture, no clothes"

    retries = 0
    while True:  # Infinite retry loop for inpainting
        try:
            safe_print("🟢 Starting inpainting process...")
            # Perform inpainting
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=30,
            ).images[0]

            # Save the result
            result.save(INPAINTED_OUTPUT_PATH)
            safe_print(
                f"✅ Inpainting completed. Output saved as '{INPAINTED_OUTPUT_PATH}'."
            )
            break  # Exit loop if inpainting is successful
        except Exception as e:
            retries += 1
            safe_print(f"❌ Inpainting failed (attempt {retries}). Error: {e}")
            time.sleep(5)  # Wait before retrying


# ======== MAIN FUNCTION ========
def main():
    try:
        # Load Image
        image = load_image(INPUT_IMAGE_PATH)

        # Load Segmentation Model
        processor, model = load_segmentation_model()

        # Preprocess image and generate clothes mask
        inputs = processor(images=image, return_tensors="pt")
        clothes_mask = generate_clothing_mask(model, inputs, image.size)

        # Save the mask
        save_black_inverted_alpha(clothes_mask, MASK_OUTPUT_PATH)

        # Retry inpainting process indefinitely
        inpaint_with_retry(INPUT_IMAGE_PATH, MASK_OUTPUT_PATH)
    except Exception as e:
        safe_print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
