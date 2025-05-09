from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import os
import sys
import torch
import json
import yaml
from typing import Optional, List, Union
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from diffusers.utils import logging
from datetime import datetime

# Add LTX-Video to path
sys.path.append(os.path.dirname(__file__))

# Import from LTX-Video
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem
from ltx_video.inference import (
    create_ltx_video_pipeline,
    load_image_to_tensor_with_resize_and_crop,
    calculate_padding,
    get_device,
    prepare_conditioning
)

# Constants
MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.get_logger("LTX-Video")

def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return 0

def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )

class ConditioningImage(BaseModel):
    image: File  # Base64 encoded image
    frame_index: Optional[int] = None  # If None, will be automatically assigned
    strength: float = 1.0  # Default to maximum strength

class AppInput(BaseAppInput):
    prompt: str
    negative_prompt: Optional[str] = "worst quality, inconsistent motion, blurry, jittery, distorted"
    height: Optional[int] = 480
    width: Optional[int] = 704
    num_frames: Optional[int] = 121
    frame_rate: Optional[int] = 25
    num_inference_steps: Optional[int] = 8  # Set to 8 for distilled model
    guidance_scale: Optional[float] = 3.0
    seed: Optional[int] = 171198
    conditioning_images: Optional[List[ConditioningImage]] = None
    offload_to_cpu: Optional[bool] = False
    image_cond_noise_scale: Optional[float] = 0.15

class AppOutput(BaseAppOutput):
    video: File

class App(BaseApp):
    async def setup(self):
        """Initialize the LTX-Video model."""
        # Load distilled config file
        config_file = "configs/ltxv-2b-0.9.6-distilled.yaml"
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        
        if not os.path.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Download model files
        self.ckpt_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video", 
            filename=self.config["checkpoint_path"], 
            repo_type='model'
        )

        self.device = get_device()
        
        # Create pipeline with config parameters
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=self.ckpt_path,
            precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config.get("sampler", "from_checkpoint"),
            device=self.device,
            enhance_prompt=self.config.get("prompt_enhancement_words_threshold", 0) > 0,
            prompt_enhancer_image_caption_model_name_or_path=self.config.get("prompt_enhancer_image_caption_model_name_or_path"),
            prompt_enhancer_llm_model_name_or_path=self.config.get("prompt_enhancer_llm_model_name_or_path"),
        )

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation with LTX-Video."""
        # Validate input dimensions
        if input_data.height > MAX_HEIGHT or input_data.width > MAX_WIDTH:
            raise ValueError(f"Dimensions exceed maximum allowed: {MAX_HEIGHT}x{MAX_WIDTH}")
        if input_data.num_frames > MAX_NUM_FRAMES:
            raise ValueError(f"Number of frames exceeds maximum allowed: {MAX_NUM_FRAMES}")

        # Handle CPU offloading
        if input_data.offload_to_cpu and not torch.cuda.is_available():
            logger.warning(
                "offload_to_cpu is set to True, but offloading will not occur since the model is already running on CPU."
            )
            offload_to_cpu = False
        else:
            offload_to_cpu = input_data.offload_to_cpu and get_total_gpu_memory() < 30

        # Set up generator with seed
        generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
        
        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((input_data.height - 1) // 32 + 1) * 32
        width_padded = ((input_data.width - 1) // 32 + 1) * 32
        num_frames_padded = ((input_data.num_frames - 2) // 8 + 1) * 8 + 1
        
        padding = calculate_padding(
            input_data.height, 
            input_data.width, 
            height_padded, 
            width_padded
        )
        
        # Prepare conditioning items
        conditioning_items = []
        
        # Handle conditioning images if provided
        if input_data.conditioning_images and len(input_data.conditioning_images) > 0:
            # Collect all images with explicit frame indices
            explicit_frames = []
            unassigned_images = []
            
            for item in input_data.conditioning_images:
                if item.frame_index is not None:
                    explicit_frames.append(item)
                else:
                    unassigned_images.append(item)
            
            # Assign frame indices to images without explicit indices
            num_unassigned = len(unassigned_images)
            if num_unassigned > 0:
                # Get list of already assigned frames
                assigned_frames = [item.frame_index for item in explicit_frames]
                
                if num_unassigned == 1:
                    # If there's only one unassigned image, place it at the middle frame
                    # or at 0 if no explicit frames yet
                    if not assigned_frames:
                        unassigned_images[0].frame_index = 0
                    else:
                        mid_frame = input_data.num_frames // 2
                        # Try to avoid collision with explicit frames
                        while mid_frame in assigned_frames:
                            mid_frame = (mid_frame + 1) % input_data.num_frames
                        unassigned_images[0].frame_index = mid_frame
                else:
                    # For multiple images, distribute them evenly across available frames
                    available_frames = [i for i in range(input_data.num_frames) if i not in assigned_frames]
                    
                    # If we have fewer available frames than images, we'll have to reuse some frames
                    if len(available_frames) < num_unassigned:
                        available_frames = list(range(input_data.num_frames))
                    
                    # Select frames at regular intervals from available frames
                    step = len(available_frames) / num_unassigned
                    for i, item in enumerate(unassigned_images):
                        frame_idx = available_frames[int(i * step)]
                        item.frame_index = frame_idx
            
            # Now process all images (both explicit and assigned)
            all_images = explicit_frames + unassigned_images
            
            # Process each conditioning image
            for item in all_images:
                # Create conditioning item
                frame_tensor = load_image_to_tensor_with_resize_and_crop(
                    item.image.path, input_data.height, input_data.width
                )
                frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
                
                # Ensure frame index is within valid range
                frame_index = max(0, min(item.frame_index, input_data.num_frames - 1))
                
                conditioning_items.append(ConditioningItem(frame_tensor, frame_index, item.strength))
        
        # Prepare input for the pipeline
        sample = {
            "prompt": input_data.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": input_data.negative_prompt,
            "negative_prompt_attention_mask": None,
        }

        # Get STG mode from config
        stg_mode = self.config.get("stg_mode", "attention_values")
        if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
            skip_layer_strategy = "attention_values"
        elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
            skip_layer_strategy = "attention_skip"
        elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
            skip_layer_strategy = "residual"
        elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
            skip_layer_strategy = "transformer_block"
        else:
            skip_layer_strategy = "attention_values"

        # Run inference with config parameters
        images = self.pipeline(
            num_inference_steps=input_data.num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=input_data.guidance_scale,
            generator=generator,
            output_type="pt",
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=input_data.frame_rate,
            **sample,
            conditioning_items=conditioning_items if conditioning_items else None,
            is_video=True,
            vae_per_channel_normalize=True,
            enhance_prompt=self.config.get("prompt_enhancement_words_threshold", 0) > 0,
            skip_layer_strategy=skip_layer_strategy,
            decode_timestep=self.config.get("decode_timestep", 0.05),
            decode_noise_scale=self.config.get("decode_noise_scale", 0.025),
            stochastic_sampling=True,  # Enable stochastic sampling for distilled model
            image_cond_noise_scale=input_data.image_cond_noise_scale,
            offload_to_cpu=offload_to_cpu
        ).images
        
        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        images = images[:, :, :input_data.num_frames, pad_top:pad_bottom, pad_left:pad_right]
        
        # Convert to video
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # Create output directory
        output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        output_filename = get_unique_filename(
            "video_output",
            ".mp4",
            prompt=input_data.prompt,
            seed=input_data.seed,
            resolution=(input_data.height, input_data.width, input_data.num_frames),
            dir=output_dir
        )
        
        # Save video
        with imageio.get_writer(output_filename, fps=input_data.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)
        
        logger.warning(f"Output saved to {output_filename}")
        return AppOutput(video=File(path=str(output_filename)))

    async def unload(self):
        """Clean up resources."""
        # Free up GPU memory
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache()