import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Enable more verbose torch logging
os.environ["TORCH_LOGS"] = "+dynamo,+distributed,+inductor"
# Enable CUDA debugging if available
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field
from PIL import Image
from diffusers import WanImageToVideoPipeline, FirstBlockCacheConfig
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline, WanTransformer3DModel, GGUFQuantizationConfig, AutoModel
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
import logging
import traceback
import psutil
import gc

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Model variants mapping for GGUF quantization from QuantStack
# Only includes variants where both HighNoise and LowNoise transformers are available
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "q2_k": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q3_k_s": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_m": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q4_k_s": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q4_k_m": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q5_0": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_1": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_k_s": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q5_k_m": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q6_k": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q8_0": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"
    }
}

DEFAULT_VARIANT = "default"

class AppInput(BaseAppInput):
    image: File = Field(description="Input image for video generation")
    prompt: str = Field(description="Text prompt for video generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    resolution: str = Field(default="720p", description="Resolution preset", enum=["480p", "720p"])
    max_area: Optional[int] = Field(default=None, description="Maximum area for image resizing (auto-set based on resolution if not specified)")
    num_frames: int = Field(default=81, description="Number of frames to generate")
    guidance_scale: float = Field(default=3.5, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=40, description="Number of denoising steps")
    fps: int = Field(default=16, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0, description="Cache threshold for transformer (0 to disable caching)")
    cache_threshold_2: float = Field(default=0, description="Cache threshold for transformer_2 (0 to disable caching)")
    
class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Image-to-Video pipeline and resources here."""
        print("Setting up Wan2.2 Image-to-Video pipeline...")
        
        # Enable comprehensive logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Enable diffusers debugging
        logging.getLogger("diffusers").setLevel(logging.DEBUG)
        logging.getLogger("transformers").setLevel(logging.DEBUG)
        logging.getLogger("accelerate").setLevel(logging.DEBUG)
        
        # Enable torch debugging
        torch.set_printoptions(profile="full")
        if hasattr(torch, 'set_warn_always'):
            torch.set_warn_always(True)
        
        print("=== SYSTEM DEBUG INFO ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        # Store resolution defaults
        self.resolution_presets = {
            "480p": {"max_area": 480 * 832},
            "720p": {"max_area": 720 * 1280}
        }
        
        # Initialize accelerator
        print("Initializing accelerator...")
        self.accelerator = Accelerator()
        
        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")
        print(f"Accelerator state: {self.accelerator.state}")
        
        # Get variant and determine if using quantization
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        
        print(f"Loading model variant: {variant}")
        
        # Load standard F16 pipeline
        print("Loading standard F16 Wan2.2 I2V pipeline...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
            torch_dtype=self.dtype
        )
        # Move to device for default F16 model
        self.pipe.enable_model_cpu_offload()
        # print(f"Moving pipeline to {self.device}...")
        # self.pipe.to(self.device)
        
        print("Setup complete!")

    def resize_image_for_pipeline(self, image: Image.Image, max_area: int) -> tuple[Image.Image, int, int]:
        """Resize image according to pipeline requirements."""
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        resized_image = image.resize((width, height))
        print(f"Resized image from {image.size} to {resized_image.size} (target area: {max_area})")
        
        return resized_image, width, height

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from image and text prompt."""
        print(f"Generating video with prompt: {input_data.prompt}")
        
        # Use resolution preset if max_area not specified
        preset = self.resolution_presets.get(input_data.resolution, self.resolution_presets["720p"])
        max_area = input_data.max_area if input_data.max_area is not None else preset["max_area"]
        print(f"Using resolution: {input_data.resolution}, max area: {max_area}")
        
        # Load and process input image
        image = Image.open(input_data.image.path).convert("RGB")
        print(f"Loaded image: {image.size}")
        
        # Resize image according to pipeline requirements
        resized_image, width, height = self.resize_image_for_pipeline(image, max_area)
        
        # Set seed if provided
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
            print(f"Using seed: {input_data.seed}")
            
        # Configure caching if thresholds are non-zero
        # First disable any existing caching to prevent conflicts
        if hasattr(self.pipe.transformer, 'disable_cache'):
            self.pipe.transformer.disable_cache()
        if hasattr(self.pipe.transformer_2, 'disable_cache'):
            self.pipe.transformer_2.disable_cache()
        
        if input_data.cache_threshold > 0:
            print(f"Enabling cache for transformer with threshold: {input_data.cache_threshold}")
            cache_config = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipe.transformer.enable_cache(cache_config)
        
        if input_data.cache_threshold_2 > 0:
            print(f"Enabling cache for transformer_2 with threshold: {input_data.cache_threshold_2}")
            cache_config_2 = FirstBlockCacheConfig(threshold=input_data.cache_threshold_2)
            self.pipe.transformer_2.enable_cache(cache_config_2)
        
        # Generate video
        print("Starting video generation...")
        output = self.pipe(
            image=resized_image,
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            height=height,
            width=width,
            num_frames=input_data.num_frames,
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
            generator=generator,
        ).frames[0]
        
        print("Video generation complete, exporting...")
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Export video
        export_to_video(output, output_path, fps=input_data.fps)
        
        print(f"Video exported to: {output_path}")
        
        return AppOutput(file=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        print("Cleaning up...")
        if hasattr(self, 'pipe'):
            del self.pipe
        
        # Clear GPU cache if using CUDA
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        print("Cleanup complete!") 