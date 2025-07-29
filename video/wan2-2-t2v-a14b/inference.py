import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.hooks import FirstBlockCacheConfig
from accelerate import Accelerator
from PIL import Image

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for video generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    resolution: str = Field(default="720p", description="Resolution preset", enum=["480p", "720p"])
    width: Optional[int] = Field(default=None, description="Width of the generated video (auto-set based on resolution if not specified)")
    height: Optional[int] = Field(default=None, description="Height of the generated video (auto-set based on resolution if not specified)")
    num_frames: int = Field(default=81, description="Number of frames to generate")
    guidance_scale: float = Field(default=4.0, description="Primary classifier-free guidance scale")
    guidance_scale_2: float = Field(default=3.0, description="Secondary guidance scale")
    num_inference_steps: int = Field(default=40, description="Number of denoising steps")
    fps: int = Field(default=16, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0, description="Cache threshold for transformer (0 to disable caching)")
    cache_threshold_2: float = Field(default=0, description="Cache threshold for transformer_2 (0 to disable caching)")

class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated file (video when num_frames > 1, image when num_frames = 1)")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 pipeline and resources here."""
        print("Setting up Wan2.2 Text-to-Video pipeline...")
        
        # Store resolution defaults
        self.resolution_presets = {
            "480p": {"width": 832, "height": 480},
            "720p": {"width": 1280, "height": 720}
        }
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")
        
        # Load VAE
        print("Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        # Load pipeline
        print("Loading Wan2.2 pipeline...")
        self.pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
            vae=self.vae, 
            torch_dtype=self.dtype
        )
        
        # Move to device
        print(f"Moving pipeline to {self.device}...")
        self.pipe.to(self.device)
        
        # Enable model offloading to reduce GPU memory usage
        print("Enabling model offloading...")
        self.pipe.enable_model_cpu_offload()
        
        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from text prompt."""
        print(f"Generating video with prompt: {input_data.prompt}")
        
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
        
        # Use resolution preset if width/height not specified
        preset = self.resolution_presets.get(input_data.resolution, self.resolution_presets["720p"])
        width = input_data.width if input_data.width is not None else preset["width"]
        height = input_data.height if input_data.height is not None else preset["height"]
        
        print(f"Using resolution: {width}x{height}")
        
        # Set seed if provided
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(input_data.seed)
        
        # Generate video
        print("Starting video generation...")
        output = self.pipe(
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            height=height,
            width=width,
            num_frames=input_data.num_frames,
            guidance_scale=input_data.guidance_scale,
            guidance_scale_2=input_data.guidance_scale_2,
            num_inference_steps=input_data.num_inference_steps,
        ).frames[0]
        
        print("Generation complete, exporting...")
        
        # Check if single frame - return image instead of video
        if input_data.num_frames == 1:
            print("Single frame detected, exporting as image...")
            
            # Create temporary file for image output
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Save the single frame as an image
            frame = output[0]  # Get the first (and only) frame
            
            # Convert tensor to numpy array
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            # Ensure values are in [0, 1] range, then convert to [0, 255] uint8
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            
            # Convert from CHW to HWC format if needed
            if len(frame.shape) == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)  # CHW -> HWC
            
            img = Image.fromarray(frame)
            img.save(output_path)
            
            print(f"Image exported to: {output_path}")
            return AppOutput(file=File(path=output_path))
        
        else:
            print("Multiple frames detected, exporting as video...")
            
            # Create temporary file for video output
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
        if hasattr(self, 'vae'):
            del self.vae
        
        # Clear GPU cache if using CUDA
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        print("Cleanup complete!")