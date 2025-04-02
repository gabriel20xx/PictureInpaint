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


# ======== LOAD INPUT IMAGE ========
def load_image_and_resize(image, target_width, target_height):
    if image is None:
        raise ValueError("No image provided.")

    try:
        # Ensure the image is in RGB format
        image = image.convert("RGB")
        print("Loaded image successfully.")
        original_width, original_height = image.size
    except Exception as e:
        raise IOError(f"Failed to process image: {e}")

    # Resize only if the original image is larger than the target dimensions
    if original_width > target_width and original_height > target_height:
        aspect_ratio = original_width / original_height

        if (target_width / target_height) > aspect_ratio:
            new_height = target_height
            new_width = int(aspect_ratio * target_height)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        return image.resize((new_width, new_height), Image.LANCZOS)

    # If the original image is smaller, return it as is
    return image


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
