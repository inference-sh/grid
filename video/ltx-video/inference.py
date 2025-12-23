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
import cv2
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

# Import from LTX-Video core functionality
from ltx_video.inference import (
    create_ltx_video_pipeline,
    calculate_padding,
    get_device,
    prepare_conditioning,
    create_latent_upsampler,
    load_media_file
)
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
import ltx_video.pipelines.crf_compressor as crf_compressor

# Constants
MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_PIXELS = MAX_HEIGHT * MAX_WIDTH  # 921,600 pixels
MAX_NUM_FRAMES = 257

logger = logging.get_logger("LTX-Video")

def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return 0

class ConditioningImage(BaseModel):
    image: File
    frame_index: Optional[int] = 0
    strength: float = 1.0

class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt to guide video generation")
    negative_prompt: Optional[str] = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to specify undesired features"
    )
    width: Optional[int] = Field(default=704, description="Width of the output video frames")
    height: Optional[int] = Field(default=480, description="Height of the output video frames")
    num_frames: Optional[int] = Field(default=121, description="Number of frames to generate")
    frame_rate: Optional[int] = Field(default=30, description="Frame rate for the output video")
    num_inference_steps: Optional[int] = Field(default=40, description="Number of denoising steps. Use 4,8,16 for distilled models")
    guidance_scale: Optional[float] = Field(default=3.0, description="Scale for classifier-free guidance")
    seed: Optional[int] = Field(default=171198, description="Random seed for reproducibility")
    conditioning_images: Optional[List[ConditioningImage]] = Field(default=None, description="List of conditioning images")
    offload_to_cpu: Optional[bool] = Field(default=False, description="Whether to offload to CPU")
    image_cond_noise_scale: Optional[float] = Field(default=0.15, description="Scale of noise for conditioning")
    input_media: Optional[File] = Field(default=None, description="Input video file for video-to-video generation")
    strength: Optional[float] = Field(default=1.0, description="Strength of input video influence")
    enable_prompt_enhancement: Optional[bool] = Field(default=None, description="Explicitly enable or disable prompt enhancement. If None, will use word count threshold logic.")
    # pipeline_config: Optional[str] = Field(
    #     default="configs/ltxv-13b-0.9.7-dev.yaml",
    #     description="Path to pipeline configuration file"
    # )

class AppOutput(BaseAppOutput):
    video: File
    
configs = {
    "2B_dev": {
        "config_file": "configs/ltxv-2b-0.9.6-dev.yaml",
    },
    "2B_distilled": {
        "config_file": "configs/ltxv-2b-0.9.6-distilled.yaml",
    },
    "13B_dev": {
        "config_file": "configs/ltxv-13b-0.9.7-dev.yaml",
    },
    "13B_distilled": {
        "config_file": "configs/ltxv-13b-0.9.7-distilled.yaml",
    }
}

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.config = None
        self.ckpt_path = None
        self.spatial_upscaler_path = None
        self.device = None
        self.latent_upsampler = None

    async def setup(self, metadata):
        """Initialize the LTX-Video model."""
        # Load config file
        self.variant_config = configs[metadata.app_variant]
        config_file = self.variant_config["config_file"]
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        
        if not os.path.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        logger.warning(f"Loaded config from {config_path}: {self.config}")

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

        # Check if prompt enhancement should be enabled
        prompt_enhancement_words_threshold = self.config.get("prompt_enhancement_words_threshold", 0)
        prompt_word_count = len(self.config.get("prompt", "").split())
        enhance_prompt = (
            prompt_enhancement_words_threshold > 0
            and prompt_word_count < prompt_enhancement_words_threshold
        )
        
        if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
            logger.info(
                f"Prompt has {prompt_word_count} words, which exceeds the threshold of {prompt_enhancement_words_threshold}. Prompt enhancement disabled."
            )
        
        # Create pipeline with config parameters
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=self.ckpt_path,
            precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config.get("sampler", "from_checkpoint"),
            device=self.device,
            enhance_prompt=enhance_prompt,
            prompt_enhancer_image_caption_model_name_or_path=self.config.get("prompt_enhancer_image_caption_model_name_or_path"),
            prompt_enhancer_llm_model_name_or_path=self.config.get("prompt_enhancer_llm_model_name_or_path"),
        )

        # Handle multi-scale pipeline
        if self.config.get("pipeline_type", None) == "multi-scale":
            if not self.spatial_upscaler_path:
                raise ValueError("Spatial upscaler model path required for multi-scale rendering")
            self.latent_upsampler = create_latent_upsampler(self.spatial_upscaler_path, self.device)
            self.pipeline = LTXMultiScalePipeline(self.pipeline, latent_upsampler=self.latent_upsampler)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run video generation with LTX-Video."""
        # Validate input dimensions
        total_pixels = input_data.height * input_data.width
        if total_pixels > MAX_PIXELS:
            raise ValueError(f"Total pixels exceed maximum allowed: {MAX_PIXELS} (current: {total_pixels})")
        if input_data.num_frames > MAX_NUM_FRAMES:
            raise ValueError(f"Number of frames exceeds maximum allowed: {MAX_NUM_FRAMES}")

        # Check if prompt enhancement should be enabled
        enhance_prompt = input_data.enable_prompt_enhancement
        if enhance_prompt is None:
            # Use word count threshold logic only if not explicitly set
            prompt_enhancement_words_threshold = self.config.get("prompt_enhancement_words_threshold", 0)
            prompt_word_count = len(input_data.prompt.split())
            enhance_prompt = (
                prompt_enhancement_words_threshold > 0
                and prompt_word_count < prompt_enhancement_words_threshold
            )
            
            if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
                logger.info(
                    f"Prompt has {prompt_word_count} words, which exceeds the threshold of {prompt_enhancement_words_threshold}. Prompt enhancement disabled."
                )
        else:
            if enhance_prompt:
                logger.info("Prompt enhancement explicitly enabled.")
            else:
                logger.info("Prompt enhancement explicitly disabled.")

        # Handle CPU offloading
        if input_data.offload_to_cpu and not torch.cuda.is_available():
            logger.warning("CPU offloading disabled - model already running on CPU")
            offload_to_cpu = False
        else:
            offload_to_cpu = input_data.offload_to_cpu and get_total_gpu_memory() < 30

        # Set up generator with seed
        generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
        
        # Adjust dimensions for padding
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
        conditioning_items = None
        if input_data.conditioning_images:
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
        config_dict = dict(self.config)
        
        # Remove non-pipeline parameters but keep pipeline_type for later
        config_dict.pop("stg_mode", None)
        config_dict.pop("checkpoint_path", None)
        config_dict.pop("spatial_upscaler_model_path", None)
        
        if stg_mode.lower() in ["stg_av", "attention_values"]:
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() in ["stg_as", "attention_skip"]:
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() in ["stg_r", "residual"]:
            skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode.lower() in ["stg_t", "transformer_block"]:
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

        # Prepare pipeline input
        sample = {
            "prompt": input_data.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": input_data.negative_prompt,
            "negative_prompt_attention_mask": None,
        }

        # Run inference
        pipeline_args = {
            "skip_layer_strategy": skip_layer_strategy,
            "generator": generator,
            "output_type": "pt",
            "callback_on_step_end": None,
            "height": height_padded,
            "width": width_padded,
            "num_frames": num_frames_padded,
            "frame_rate": input_data.frame_rate,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            **sample,
            "media_items": media_item,
            "strength": input_data.strength,
            "conditioning_items": conditioning_items,
            "is_video": True,
            "vae_per_channel_normalize": True,
            "image_cond_noise_scale": input_data.image_cond_noise_scale,
            "mixed_precision": (self.config["precision"] == "mixed_precision"),
            "offload_to_cpu": offload_to_cpu,
            "device": self.device,
            "enhance_prompt": enhance_prompt,
        }

        if self.config.get("pipeline_type", None) == "multi-scale":
            pipeline_args.update({
                "downscale_factor": float(self.config["downscale_factor"]),
                "first_pass": self.config["first_pass"],
                "second_pass": self.config["second_pass"],
            })
            logger.warning(f"Adding multi-scale args with downscale_factor={self.config['downscale_factor']}")

        logger.warning(f"Final pipeline args: {pipeline_args}")
        images = self.pipeline(**pipeline_args).images

        # Process output
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        images = images[:, :, :input_data.num_frames, pad_top:pad_bottom, pad_left:pad_right]

        # Create output directory
        output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save output
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        output_filename = str(output_dir / f"video_output_{input_data.seed}.mp4")
        with imageio.get_writer(output_filename, fps=input_data.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)

        logger.warning(f"Output saved to {output_filename}")
        return AppOutput(video=File(path=output_filename))

