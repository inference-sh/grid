from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import yaml
from typing import Optional, List
from pathlib import Path
import imageio
import numpy as np
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
from diffusers.utils import logging
from datetime import datetime
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.pipelines.pipeline_ltx_video import (
    LTXMultiScalePipeline
)

# Add LTX-Video to path
sys.path.append(os.path.dirname(__file__))

# Import from LTX-Video
from ltx_video.inference import (
    create_ltx_video_pipeline,
    calculate_padding,
    get_device,
    prepare_conditioning,
    create_latent_upsampler,
    load_media_file
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
    prompt: str = Field(description="Text prompt to guide video generation")
    negative_prompt: Optional[str] = Field(default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt to specify undesired features")
    width: Optional[int] = Field(default=704, description="Width of the output video frames")
    height: Optional[int] = Field(default=480, description="Height of the output video frames")
    num_frames: Optional[int] = Field(default=121, description="Number of frames to generate in the output video")
    frame_rate: Optional[int] = Field(default=25, description="Frame rate for the output video")
    num_inference_steps: Optional[int] = Field(default=40, description="Number of denoising steps in the diffusion process")
    guidance_scale: Optional[float] = Field(default=3.0, description="Scale for classifier-free guidance")
    seed: Optional[int] = Field(default=171198, description="Random seed for reproducibility")
    conditioning_images: Optional[List[ConditioningImage]] = Field(default=None, description="List of conditioning images with their parameters")
    offload_to_cpu: Optional[bool] = Field(default=False, description="Whether to offload unnecessary computations to CPU")
    image_cond_noise_scale: Optional[float] = Field(default=0.15, description="Scale of noise to add to conditioning images")
    input_media: Optional[File] = Field(default=None, description="Input video file for video-to-video generation")
    strength: Optional[float] = Field(default=1.0, description="Strength of input video influence in video-to-video generation")
    pipeline_config: Optional[str] = Field(default="configs/ltxv-2b-0.9.6-dev.yaml", description="Path to custom pipeline configuration file")

class AppOutput(BaseAppOutput):
    video: File

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.config = None
        self.ckpt_path = None
        self.spatial_upscaler_path = None
        self.device = None
        self.latent_upsampler = None

    async def setup(self):
        """Initialize the LTX-Video model."""
        # Load dev config file
        config_file = "configs/ltxv-2b-0.9.6-dev.yaml"
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

        if self.config.get("spatial_upscaler_model_path"):
            self.spatial_upscaler_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=self.config["spatial_upscaler_model_path"],
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

        # Handle multi-scale pipeline if configured
        if self.config.get("pipeline_type", None) == "multi-scale":
            if not self.spatial_upscaler_path:
                raise ValueError(
                    "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
                )
            self.latent_upsampler = create_latent_upsampler(
                self.spatial_upscaler_path, self.device
            )
            self.pipeline = LTXMultiScalePipeline(self.pipeline, latent_upsampler=self.latent_upsampler)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation with LTX-Video."""
        # Load custom config if provided
        if input_data.pipeline_config and input_data.pipeline_config != self.config.get("config_file"):
            config_path = os.path.join(os.path.dirname(__file__), input_data.pipeline_config)
            if not os.path.exists(config_path):
                raise ValueError(f"Custom config file {config_path} does not exist")
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

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

        # Handle input media for video-to-video
        media_item = None
        if input_data.input_media:
            media_item = load_media_file(
                media_path=input_data.input_media.path,
                height=input_data.height,
                width=input_data.width,
                max_frames=num_frames_padded,
                padding=padding,
            )
        
        # Prepare conditioning items
        conditioning_items = []
        if input_data.conditioning_images:
            # Convert ConditioningImage objects to the format expected by prepare_conditioning
            media_paths = [img.image.path for img in input_data.conditioning_images]
            strengths = [img.strength for img in input_data.conditioning_images]
            start_frames = [img.frame_index for img in input_data.conditioning_images]
            
            conditioning_items = prepare_conditioning(
                conditioning_media_paths=media_paths,
                conditioning_strengths=strengths,
                conditioning_start_frames=start_frames,
                height=input_data.height,
                width=input_data.width,
                num_frames=input_data.num_frames,
                padding=padding,
                pipeline=self.pipeline,
            )

        # Get STG mode from config
        stg_mode = self.config.get("stg_mode", "attention_values")
        if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
            skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

        # Prepare input for the pipeline
        sample = {
            "prompt": input_data.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": input_data.negative_prompt,
            "negative_prompt_attention_mask": None,
        }

        # Run inference
        images = self.pipeline(
            **self.config,
            skip_layer_strategy=skip_layer_strategy,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=input_data.frame_rate,
            **sample,
            media_items=media_item,
            strength=input_data.strength,
            conditioning_items=conditioning_items,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=input_data.image_cond_noise_scale,
            mixed_precision=(self.config["precision"] == "mixed_precision"),
            offload_to_cpu=offload_to_cpu,
            device=self.device,
            enhance_prompt=self.config.get("prompt_enhancement_words_threshold", 0) > 0,
        ).images

        # Process output
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        images = images[:, :, :input_data.num_frames, pad_top:pad_bottom, pad_left:pad_right]

        # Create output directory
        output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save output
        for i in range(images.shape[0]):
            video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
            video_np = (video_np * 255).astype(np.uint8)
            
            if video_np.shape[0] == 1:
                output_filename = get_unique_filename(
                    f"image_output_{i}",
                    ".png",
                    prompt=input_data.prompt,
                    seed=input_data.seed,
                    resolution=(input_data.height, input_data.width, input_data.num_frames),
                    dir=output_dir,
                )
                imageio.imwrite(output_filename, video_np[0])
            else:
                output_filename = get_unique_filename(
                    f"video_output_{i}",
                    ".mp4",
                    prompt=input_data.prompt,
                    seed=input_data.seed,
                    resolution=(input_data.height, input_data.width, input_data.num_frames),
                    dir=output_dir,
                )
                with imageio.get_writer(output_filename, fps=input_data.frame_rate) as video:
                    for frame in video_np:
                        video.append_data(frame)

            logger.warning(f"Output saved to {output_filename}")
            return AppOutput(video=File(path=str(output_filename)))

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if hasattr(self, 'latent_upsampler'):
            del self.latent_upsampler
        torch.cuda.empty_cache()