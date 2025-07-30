import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils import export_to_video
from diffusers.hooks import FirstBlockCacheConfig
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from PIL import Image

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Model variants mapping for GGUF quantization from QuantStack
# Only includes variants where both HighNoise and LowNoise transformers are available
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "q2_k": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q3_k_s": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_m": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q4_0": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_0.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_0.gguf"
    },
    "q4_1": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_1.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_1.gguf"
    },
    "q4_k_s": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q4_k_m": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q5_0": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_1": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_k_s": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q5_k_m": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q6_k": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q8_0": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"
    }
}

DEFAULT_VARIANT = "default"

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
        
        # Get variant and determine if using quantization
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            print(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        
        print(f"Loading model variant: {variant}")
        
        # Load VAE
        print("Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        if variant == "default":
            # Load standard F16 pipeline
            print("Loading standard F16 Wan2.2 T2V pipeline...")
            self.pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
                vae=self.vae, 
                torch_dtype=self.dtype
            )
            # Move to device and enable model offloading
            print(f"Moving pipeline to {self.device}...")
            self.pipe.enable_model_cpu_offload()
        else:
            # Load quantized transformers
            print(f"Loading quantized transformers for {variant}...")
            repo_id = "QuantStack/Wan2.2-T2V-A14B-GGUF"
            variant_files = MODEL_VARIANTS[variant]
            
            high_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['high_noise'])
           
            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )

            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['low_noise'])
            
            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )
            
            self.pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                vae=self.vae,
                transformer=transformer_high_noise,  # High noise goes to main transformer
                transformer_2=transformer_low_noise,  # Low noise goes to transformer_2
                torch_dtype=self.dtype
            )
             
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