import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field
from PIL import Image
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
from accelerate import Accelerator

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    image: File = Field(description="Input image for video generation")
    prompt: str = Field(description="Text prompt for video generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    resolution: str = Field(default="720p", description="Resolution preset", enum=["480p", "720p"])
    max_area: Optional[int] = Field(default=None, description="Maximum area for image resizing (auto-set based on resolution if not specified)")
    num_frames: int = Field(default=121, description="Number of frames to generate")
    guidance_scale: float = Field(default=5.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=50, description="Number of denoising steps")
    fps: int = Field(default=24, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0.1, description="First Block Cache threshold (higher = more aggressive caching)")

class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2-TI2V-5B Image-to-Video pipeline and resources here."""
        print("Setting up Wan2.2-TI2V-5B Image-to-Video pipeline...")
        
        # Store resolution defaults (using TI2V resolution standards)
        self.resolution_presets = {
            "480p": {"max_area": 480 * 832},
            "720p": {"max_area": 704 * 1280}  # Updated to TI2V's 704 height
        }
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")
        
        # Model ID for the TI2V 5B variant (using I2V pipeline with TI2V model)
        self.model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        # Load pipeline
        print("Loading Wan2.2-TI2V-5B I2V pipeline...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=self.dtype
        )
        
        # Move to device
        print(f"Moving pipeline to {self.device}...")
        self.pipe.to(self.device)
        
        # # Enable CPU offloading to save GPU memory
        # print("Enabling CPU offloading...")
        # self.pipe.enable_model_cpu_offload()
        
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
        
        if input_data.cache_threshold > 0:
            print(f"Enabling cache for transformer with threshold: {input_data.cache_threshold}")
            cache_config = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipe.transformer.enable_cache(cache_config)
        
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