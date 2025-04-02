from transformers import (
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
)
import torch
import cv2
import numpy as np
from datetime import datetime
import os
from PIL import Image


# ======== LOAD SEGMENTATION MODEL ========
def load_segmentation_model():
    try:
        print("Loading clothes segmentation model...")
        processor = SegformerImageProcessor.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        )
        model.eval()
        print("Segmentation model loaded.")
        return processor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load segmentation model: {e}")


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
        raise ValueError("No clothing detected in the image.")

    return clothes_mask_resized


# ======== APPLY MASK GROW AND SAVE ========
def save_black_inverted_alpha(clothes_mask, output_path, mask_grow_pixels=15):
    dilate_size = max(5, mask_grow_pixels)  # Ensure mask grows sufficiently
    close_size = max(3, mask_grow_pixels // 3)  # Ensure minimum size of 3

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )

    # --- Expand the mask slightly to remove unwanted artifacts ---
    mask = cv2.dilate(clothes_mask, dilate_kernel, iterations=1)

    # --- Smooth edges to avoid harsh transitions ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Apply a Gaussian blur for extra smoothing
    blur_ksize = max(5, (mask_grow_pixels // 2) * 2 + 1)  # Ensure odd kernel size
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), sigmaX=0, sigmaY=0)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"mask_{timestamp}.png"

    # Define output directory
    output_dir = "output/masks"

    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    # Construct full output path
    output_path = os.path.join(output_dir, filename)

    # Save as grayscale PNG
    Image.fromarray(mask).save(output_path)
    print(f"Mask saved: {output_path}")
    return output_path, mask


def generate_mask(input_image, mask_grow_pixels):
    print("Starting...")
    processor, segmentation_model = load_segmentation_model()
    mask = generate_clothing_mask(segmentation_model, processor, input_image)

    mask_path, mask = save_black_inverted_alpha(mask, mask_grow_pixels)


# ======== MAIN FUNCTION ========
if __name__ == "__main__":
    input_image_path = "input/image.jpg"  # Replace with your image path
    mask_grow_pixels = 15  # Adjust as needed

    input_image = Image.open(input_image_path)
    generate_mask(input_image, mask_grow_pixels)
    print("Process completed.")
