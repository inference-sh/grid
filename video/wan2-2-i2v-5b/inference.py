import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
import PIL
from typing import Optional, Union, List
from pydantic import Field
from PIL import Image
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, ModularPipeline, GGUFQuantizationConfig, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.hooks import FirstBlockCacheConfig
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
import imageio

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Model variants mapping for GGUF quantization from QuantStack
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "q2_k": "Wan2.2-TI2V-5B-Q2_K.gguf",
    "q3_k_s": "Wan2.2-TI2V-5B-Q3_K_S.gguf",
    "q3_k_m": "Wan2.2-TI2V-5B-Q3_K_M.gguf",
    "q4_0": "Wan2.2-TI2V-5B-Q4_0.gguf",
    "q4_1": "Wan2.2-TI2V-5B-Q4_1.gguf",
    "q4_k_s": "Wan2.2-TI2V-5B-Q4_K_S.gguf",
    "q4_k_m": "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    "q5_0": "Wan2.2-TI2V-5B-Q5_0.gguf",
    "q5_1": "Wan2.2-TI2V-5B-Q5_1.gguf",
    "q5_k_s": "Wan2.2-TI2V-5B-Q5_K_S.gguf",
    "q5_k_m": "Wan2.2-TI2V-5B-Q5_K_M.gguf",
    "q6_k": "Wan2.2-TI2V-5B-Q6_K.gguf",
    "q8_0": "Wan2.2-TI2V-5B-Q8_0.gguf"
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
    num_frames: int = Field(default=121, description="Number of frames to generate")
    guidance_scale: float = Field(default=5.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=50, description="Number of denoising steps")
    fps: int = Field(default=24, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0.1, description="First Block Cache threshold (higher = more aggressive caching)")
    video_output_quality: int = Field(default=5, ge=1, le=9, description="Video output quality (1-9)")

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
        
        print(f"Metadata: {metadata}")
        # Get variant and determine if using quantization
        variant = metadata.get("app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            print(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        
        print(f"Loading model variant: {variant}")
        
        # Model ID for the TI2V 5B variant (using I2V pipeline with TI2V model)
        self.model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        
        # use default wan image processor to resize and crop the image
        self.image_processor = ModularPipeline.from_pretrained("YiYiXu/WanImageProcessor", trust_remote_code=True)

        if variant == "default":
            # Load standard F16 pipeline
            print("Loading standard F16 Wan2.2-TI2V-5B I2V pipeline...")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=self.dtype,
                vae=self.vae,
            )
            # Move to device and enable model offloading
            print(f"Moving pipeline to {self.device}...")
            self.pipe.enable_model_cpu_offload()
        else:
            # Load quantized transformer
            print(f"Loading quantized transformer for {variant}...")
            repo_id = "QuantStack/Wan2.2-TI2V-5B-GGUF"
            model_file = MODEL_VARIANTS[variant]
            
            model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
            
            transformer = WanTransformer3DModel.from_single_file(
                model_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config=self.model_id,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                vae=self.vae,
                transformer=transformer,
                torch_dtype=self.dtype
            )
             
            self.pipe.enable_model_cpu_offload()
        
        print("Setup complete!")

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
        resized_image = self.image_processor(
            image=input_data.image.path,
            max_area=max_area, output="processed_image")
        
        width = resized_image.width
        height = resized_image.height
        
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
        export_to_video(output, output_path, fps=input_data.fps, quality=input_data.video_output_quality)
        
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