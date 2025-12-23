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
from .pipeline_i2v_wan_infinite import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline, WanTransformer3DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
import logging
import gc

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from contextlib import contextmanager

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
    image: File = Field(description="Input image for midpoint conditioning the generation")
    prompt: str = Field(description="Text prompt for video generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    resolution: str = Field(default="720p", description="Resolution preset", enum=["480p", "720p"])
    num_frames: int = Field(default=81, description="Number of frames to generate")
    guidance_scale: float = Field(default=3.5, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=40, description="Number of denoising steps")
    fps: int = Field(default=16, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    boundary_ratio: float = Field(default=0.875, ge=0.0, le=1.0, description="Boundary ratio for dual transformer setup (0-1)")
    end_image: Optional[File] = Field(default=None, description="Optional end image for first-to-last frame video generation")
    dual_continuous_generation: bool = Field(default=False, description="Generate dual continuous video sequences with overlapping conditioning")
    overlap_frames: int = Field(default=16, ge=4, le=32, description="Number of overlapping frames between dual sequences")

class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Image-to-Video pipeline and resources here."""
        # Store resolution defaults
        self.resolution_presets = {
            "480p": {"max_area": 480 * 832},
            "720p": {"max_area": 720 * 1280}
        }
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Get variant and determine if using quantization/offloading
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        # Offloading policy: default always uses CPU offload; explicit lowvram uses group offload
        use_group_offload = variant.endswith("_offload_lowvram")
        # Strip suffix for base variant lookup
        base_variant = variant.replace("_offload_lowvram", "").replace("_offload", "")
        if base_variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            base_variant = DEFAULT_VARIANT
        
        # Load pipeline based on variant
        if base_variant == "default":
            # Load standard F16 pipeline
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                #"Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
                "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
                torch_dtype=self.dtype
            )
            # Apply offloading/device placement
            if use_group_offload:
                onload_device = self.device
                offload_device = torch.device("cpu")
                # Group offloading on key modules
                self.pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                self.pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                if hasattr(self.pipe, 'transformer_2'):
                    self.pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            else:
                # Default: enable CPU offload for memory efficiency
                self.pipe.enable_model_cpu_offload()
        else:
            # Load quantized transformers
            repo_id = "QuantStack/Wan2.2-I2V-A14B-GGUF"
            variant_files = MODEL_VARIANTS[base_variant]
            
            # Download and load high noise transformer (main transformer)
            high_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['high_noise'])
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            
            # Download and load low noise transformer (transformer_2)
            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['low_noise'])
            
            # Force garbage collection again
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )
            
            # Load pipeline with quantized transformers
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
                transformer=transformer_high_noise,  # High noise goes to main transformer
                transformer_2=transformer_low_noise,  # Low noise goes to transformer_2
                torch_dtype=self.dtype
            )
            # Apply offloading/device placement
            if use_group_offload:
                onload_device = self.device
                offload_device = torch.device("cpu")
                self.pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                self.pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                if hasattr(self.pipe, 'transformer_2'):
                    self.pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            else:
                # Default: enable CPU offload for memory efficiency
                self.pipe.enable_model_cpu_offload()
        
        #print("Compiling transformers...")
        #self.pipe.transformer.compile_repeated_blocks(fullgraph=True)
        #self.pipe.transformer_2.compile_repeated_blocks(fullgraph=True)

        print("Setup complete!")

    def resize_image_for_pipeline(self, image: Image.Image, max_area: int) -> tuple[Image.Image, int, int]:
        """Resize image according to pipeline requirements."""
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        resized_image = image.resize((width, height))
        
        return resized_image, width, height

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from input image with midpoint conditioning."""
        # Use resolution preset
        preset = self.resolution_presets.get(input_data.resolution, self.resolution_presets["720p"])
        max_area = preset["max_area"]

        # Register boundary ratio to pipeline config
        try:
            self.pipe.register_to_config(boundary_ratio=input_data.boundary_ratio)
        except Exception:
            pass

        # Load and process input image
        with Image.open(input_data.image.path) as pil_image:
            pil_image = pil_image.convert("RGB")
            resized_image, width, height = self.resize_image_for_pipeline(pil_image, max_area)

        # Handle end_image if provided
        last_image = None
        if input_data.end_image is not None:
            with Image.open(input_data.end_image.path) as pil_end_image:
                pil_end_image = pil_end_image.convert("RGB")
                last_image, _, _ = self.resize_image_for_pipeline(pil_end_image, max_area)

        # Set seed if provided
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(input_data.seed)

        # Generate video with midpoint conditioning
        try:
            with torch.inference_mode():
                output = self.pipe(
                    image=resized_image,
                    initial_images=None,  # No additional initial images for this simple version
                    midpoint_conditioning=True,  # Always use midpoint conditioning
                    dual_continuous_generation=input_data.dual_continuous_generation,
                    overlap_frames=input_data.overlap_frames,
                    prompt=input_data.prompt,
                    negative_prompt=input_data.negative_prompt,
                    height=height,
                    width=width,
                    num_frames=input_data.num_frames,
                    guidance_scale=input_data.guidance_scale,
                    num_inference_steps=input_data.num_inference_steps,
                    generator=generator,
                    last_image=last_image,
                )
                
                # Handle output based on generation mode
                if input_data.dual_continuous_generation:
                    frames = output.frames  # Dual mode returns full frame list
                else:
                    frames = output.frames[0]  # Single mode returns batched frames

            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                output_path = temp_file.name

            # Export video
            export_to_video(frames, output_path, fps=input_data.fps)

            return AppOutput(file=File(path=output_path))
            
        finally:
            try:
                del resized_image
                del last_image
                del generator
                del output
                del frames
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    async def unload(self):
        """Clean up resources here."""
        if hasattr(self, 'pipe'):
            del self.pipe
        
        # Clear GPU cache if using CUDA
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache() 