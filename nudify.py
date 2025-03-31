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
from huggingface_hub import login
from safetensors.torch import load_file
from PIL import Image
from datetime import datetime
import gradio as gr
import requests
import os
import xformers
import numpy as np
import cv2
import logging
import sys
import shutil
import time

# Use python 3.11 for this script

# Run these once
# python -m venv fluxfill
# Activate venv on CMD: fluxfill\Scripts\activate
# Activate venv on PS: fluxfill\Scripts\Activate.ps1
# Activate venv on Linux: source fluxfill/bin/activate
# pip install torch==2.1.0 torchvision==0.15.2 torchaudio==2.0.2 xformers --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers pillow numpy<2 opencv-python accelerate sentencepiece peft protobuf gradio triton
# pip install https://github.com/bycloud-AI/DiffBIR-Windows/raw/refs/heads/main/triton-2.0.0-cp310-cp310-win_amd64.whl
# set PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True,max_split_size_mb=64,garbage_collection_threshold=0.98

# ======== CONFIGURATION ========
MASK_OUTPUT_PATH = "output/masks"
MASK_FILENAME = "mask_output.png"
INPAINTED_OUTPUT_PATH = "output/images"
OUTPUT_FILENAME = "nudified_output.png"
AVAILABLE_CHECKPOINTS = ["black-forest-labs/FLUX.1-Fill-dev"]
AVAILABLE_LORAS = ["None", "xey/sldr_flux_nsfw_v2-studio"]
LORA_MODEL_IDS = ["None", "1392143"]
INVERT_SIGMAS = False
USE_KARRAS_SIGMAS = False
USE_EXPONENTIAL_SIGMAS = False
USE_BETA_SIGMAS = True
PROMPT = "naked body, realistic skin texture, no clothes, nude, no bra, no top, no panties, no pants, no shorts. \
Remove the cloth from the image."
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5  # Default guidance scale (adjustable)
SAMPLER_NAME = "Euler"  # Change this to the desired sampler
MASK_GROW_PIXELS = 15  # Amount to grow (dilate) mask
TARGET_WIDTH = 2048
TARGET_HEIGHT = 2048

# Read HF_TOKEN from the environment
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("Python received HF_TOKEN successfully.")
    login(token=hf_token)
else:
    print("Python: HF_TOKEN is not set!")

# Ensure the directory exists
os.makedirs("output/logs", exist_ok=True)

logging.basicConfig(
    filename="output/logs/output.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


class LoggerWriter:
    def write(self, message):
        if message.strip():  # Avoid writing empty lines
            logging.info(message.strip())

    def flush(self):
        pass  # No need to implement flush for logging

    def isatty(self):  # Fix for uvicorn expecting isatty()
        return False


sys.stdout = LoggerWriter()

print("This will be logged instead of printed")


def get_system_information():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device:", torch.cuda.get_device_name(0))  # Should show GTX 1080
        print("Total memory:", torch.cuda.get_device_properties(0).total_memory)
    print("xFormers version:", xformers.__version__)


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
def save_black_inverted_alpha(clothes_mask, mask_grow_pixels=15):
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


def apply_lora(pipe, lora_model_id):
    # LoRA Model ID or URL from CivitAI
    save_dir = "lora_models"

    # Step 1: Fetch the LoRA model details from CivitAI API
    response = requests.get(f"https://civitai.com/api/v1/models/{lora_model_id}")
    data = response.json()

    # Step 2: Find the latest SafeTensors file
    model_file_url = None
    model_filename = None

    for version in data.get("modelVersions", []):
        for file in version.get("files", []):
            if file["name"].endswith(".safetensors"):  # Prioritize SafeTensors format
                model_file_url = file["downloadUrl"]
                model_filename = file["name"]
                break
        if model_file_url:
            break

    if model_file_url and model_filename:
        # Step 3: Download the LoRA model
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_filename)

        if not os.path.exists(save_path):  # Avoid re-downloading
            print(f"Downloading {model_filename}...")
            with requests.get(model_file_url, stream=True) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"LoRA downloaded to {save_path}")
        else:
            print(f"LoRA already exists: {save_path}")

        print(f"Loading LoRA: {save_path}")
        state_dict = load_file(save_path)
        pipe.load_lora_weights(state_dict)
        pipe.to("cuda")  # Move to GPU if available

        print(f"LoRA {model_filename} applied successfully.")
    else:
        print("No LoRA file found for this model.")
    return pipe


def apply_scheduler(
    pipe,
    sampler_name,
    invert_sigmas,
    use_karras_sigmas,
    use_exponential_sigmas,
    use_beta_sigmas,
):
    # Set the sampler (scheduler)
    print(f"Setting scheduler {sampler_name}...")
    pipe.scheduler = get_scheduler(sampler_name, pipe.scheduler)
    print("Scheduler set")

    # Enable sigmas
    pipe.scheduler.config["use_karras_sigmas"] = use_karras_sigmas
    pipe.scheduler.config["use_exponential_sigmas"] = use_exponential_sigmas
    pipe.scheduler.config["use_beta_sigmas"] = use_beta_sigmas

    # Optionally, also adjust other parameters if needed
    pipe.scheduler.config["invert_sigmas"] = (
        invert_sigmas  # Leave as False or adjust as needed
    )

    print(f"Config: {pipe.scheduler.config}")
    return pipe


def get_device():
    # Load model with optimized settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force CPU if needed
    if (
        device == "cuda"
        and torch.cuda.get_device_properties(0).total_memory < 6 * 1024 * 1024 * 1024
    ):
        print("Low VRAM detected, switching to CPU mode.")
        device = "cpu"

    print(f"Using device: {device}")

    return device


def load_pipeline(model):
    device = get_device()

    # Force minimal RAM/VRAM usage
    if device == "cuda":
        torch.cuda.empty_cache()  # Free GPU memory
        print("VRAM cache cleared")

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the Flux model
    print(f"Loading Flux model {model}...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")  # Adjust if needed
    while True:
        try:
            pipe = FluxFillPipeline.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                local_files_only=True,
                offload_folder="./offload_cache",
            )
            print("✅ Model loaded from local cache.")
            return pipe
        except OSError:
            print(
                "⚠️ Model not found or files are corrupt. Deleting local files and redownloading..."
            )

            # Delete cached files if they exist
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print("🗑️ Cache cleared.")

            try:
                pipe = FluxFillPipeline.from_pretrained(
                    model,
                    torch_dtype=torch_dtype,
                )
                print("✅ Model downloaded and saved to cache.")
                return pipe
            except Exception as e:
                print(f"❌ Download failed: {e}")
                if "timeout" in str(e).lower():
                    print("⏳ Retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    break  # Exit loop for non-timeout errors

        raise RuntimeError("🚨 Failed to load the model after multiple attempts.")

    pipe.to(device)
    print(f"FluxFillPipeline loaded on {device}.")
    return pipe


# ======== INPAINTING FUNCTION ========
def inpaint(
    image,
    mask,
    pipe,
    prompt,
    num_inference_steps,
    guidance_scale,
):
    device = get_device()
    pipe.to(device)

    # Free memory after loading
    if device == "cuda":
        torch.cuda.empty_cache()
        print("VRAM cache cleared")
        pipe.enable_xformers_memory_efficient_attention()  # ✅ Requires `pip install xformers`
        print("Enabled xFormers optimization.")
        torch.backends.cuda.matmul.allow_tf32 = True
        print("Matmul allow tf32 enabled.")

    try:
        print(f"Starting inpainting process on {device}...")
        with torch.no_grad():  # Prevents storing intermediate gradients
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        return result
    except Exception as e:
        print(f"Inpainting failed. Error: {e}")


def save_result(result, image):
    # Ensure same size output
    result = result.resize(image.size)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"mask_{timestamp}.png"

    # Define output directory
    output_dir = "output/images"

    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    # Construct full output path
    output_path = os.path.join(output_dir, filename)

    result.save(output_path)
    print(f"Inpainting completed. Output saved as '{output_path}'.")
    return output_path


def generate_mask(input_image, mask_grow_pixels):
    print("Starting...")
    image = load_image_and_resize(input_image, TARGET_WIDTH, TARGET_HEIGHT)
    processor, segmentation_model = load_segmentation_model()
    mask = generate_clothing_mask(segmentation_model, processor, image)

    mask_path, mask = save_black_inverted_alpha(mask, mask_grow_pixels)

    return mask_path, mask, image


# ======== MAIN FUNCTION ========
def process_image(
    image,
    mask,
    prompt,
    checkpoint_model,
    lora_model_id,
    num_inference_steps,
    guidance_scale,
    sampler_name,
    use_karras_sigmas,
    use_exponential_sigmas,
    use_beta_sigmas,
    invert_sigmas,
):
    try:
        pipe = load_pipeline(checkpoint_model)

        if lora_model_id != "None":
            pipe = apply_lora(pipe, lora_model_id)

        pipe = apply_scheduler(
            pipe,
            sampler_name,
            invert_sigmas,
            use_karras_sigmas,
            use_exponential_sigmas,
            use_beta_sigmas,
        )

        if isinstance(image, np.ndarray):  # If stored as NumPy array
            img = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, str):  # If stored as file path
            img = Image.open(image)
        elif isinstance(image, Image.Image):  # Already PIL Image
            img = image
        else:
            raise ValueError("Unsupported image format received.")

        # Retry inpainting process indefinitely with guidance scale
        result = inpaint(
            img,
            mask,
            pipe,
            prompt,
            num_inference_steps,
            guidance_scale,
        )

        output_path = save_result(result, image)
        return output_path
    except Exception as e:
        print(f"Error: {e}")


# Define Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Fully Configurable Hugging Face Inpainting")
    gr.Markdown(
        "## Select a checkpoint and LoRA from Hugging Face and apply inpainting."
    )
    submit_button = gr.Button("Submit", elem_id="submit_button")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image", height=400)
            mask_output = gr.Image(
                label="Generated Mask", elem_id="mask_output", height=400
            )
            final_output = gr.Image(
                label="Final Image", elem_id="final_output", height=400
            )

        with gr.Column():
            prompt_input = gr.Textbox(
                value=PROMPT, placeholder="Prompt", label="Prompt"
            )
            checkpoint_input = gr.Dropdown(
                choices=AVAILABLE_CHECKPOINTS,
                value=AVAILABLE_CHECKPOINTS[0],
                label="Checkpoint Model",
            )
            lora_input = gr.Dropdown(
                choices=LORA_MODEL_IDS, value=LORA_MODEL_IDS[1], label="LoRA Model"
            )
            steps_input = gr.Slider(5, 50, value=25, step=1, label="Inference Steps")
            guidance_input = gr.Slider(
                1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale"
            )
            mask_grow_pixels_input = gr.Slider(
                0, 50, value=MASK_GROW_PIXELS, step=1, label="Mask Growth (px)"
            )
            sampler_input = gr.Dropdown(
                choices=["Euler", "DPM++ 2M"], value=SAMPLER_NAME, label="Sampler Type"
            )
            use_karras_sigmas_input = gr.Checkbox(
                value=USE_KARRAS_SIGMAS, label="Use Karras Sigmas"
            )
            use_exponential_sigmas_input = gr.Checkbox(
                value=USE_EXPONENTIAL_SIGMAS, label="Use Exponential Sigmas"
            )
            use_beta_sigmas_input = gr.Checkbox(
                value=USE_BETA_SIGMAS, label="Use Beta Sigmas"
            )
            invert_sigmas_input = gr.Checkbox(
                value=INVERT_SIGMAS, label="Invert Sigmas"
            )

            mask = gr.State()  # Stores intermediate data
            image = gr.State()  # Stores intermediate data

    submit_button.click(
        fn=generate_mask,
        inputs=[
            image_input,
            mask_grow_pixels_input,
        ],
        outputs=[mask_output, mask, image],
    )

    # Step 2: Generate second image using stored value
    mask_output.change(
        process_image,
        inputs=[
            image,
            mask,
            prompt_input,
            checkpoint_input,
            lora_input,
            steps_input,
            guidance_input,
            sampler_input,
            use_karras_sigmas_input,
            use_exponential_sigmas_input,
            use_beta_sigmas_input,
            invert_sigmas_input,
        ],
        outputs=final_output,
    )


if __name__ == "__main__":
    get_system_information()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        debug=True,
        show_api=False,
    )
    print("Gradio app is running and ready to use at http://127.0.0.1:7860")
