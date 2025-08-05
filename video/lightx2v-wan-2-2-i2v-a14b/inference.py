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
from diffusers import WanImageToVideoPipeline, FirstBlockCacheConfig, ModularPipeline, GGUFQuantizationConfig, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
import safetensors.torch

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Model variants mapping for GGUF quantization from QuantStack
# Only includes variants where both HighNoise and LowNoise transformers are available
MODEL_VARIANTS = {
    # 480p variants
    "default_480p": None,  # Use default F16 model for 480p
    "default_480p_offload": None,  # Use default F16 model with offloading for 480p
    "fp16_480p": None,  # Use default F16 model with CPU offload for 480p
    "fp16_480p_offload": None,  # Use default F16 model with CPU offload and group offloading for 480p
    "q2_k_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q2_k_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q3_k_s_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_s_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_m_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q3_k_m_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q4_k_s_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q4_k_s_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q4_k_m_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q4_k_m_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q5_0_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_0_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_1_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_1_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_k_s_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q5_k_s_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q5_k_m_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q5_k_m_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q6_k_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q6_k_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q8_0_480p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"
    },
    "q8_0_480p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"
    },
    
    # 720p variants
    "default_720p": None,  # Use default F16 model for 720p
    "default_720p_offload": None,  # Use default F16 model with offloading for 720p
    "fp16_720p": None,  # Use default F16 model with CPU offload for 720p
    "fp16_720p_offload": None,  # Use default F16 model with CPU offload and group offloading for 720p
    "q2_k_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q2_k_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q3_k_s_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_s_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_m_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q3_k_m_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q4_k_s_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q4_k_s_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q4_k_m_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q4_k_m_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q5_0_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_0_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_1_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_1_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_k_s_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q5_k_s_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q5_k_m_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q5_k_m_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q6_k_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q6_k_720p_offload": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q8_0_720p": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"
    },
    "q8_0_720p_offload": {
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
    resolution: str = Field(default=None, description="Resolution preset (auto-detected from variant if not specified)", enum=["480p", "720p"])
    max_area: Optional[int] = Field(default=None, description="Maximum area for image resizing (auto-set based on resolution if not specified)")
    num_frames: int = Field(default=81, description="Number of frames to generate")
    guidance_scale: float = Field(default=1.0, description="Classifier-free guidance scale (optimized for LightX2V)")
    num_inference_steps: int = Field(default=4, description="Number of denoising steps (optimized for LightX2V)")
    fps: int = Field(default=16, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0, description="Cache threshold for transformer (0 to disable caching)")
    cache_threshold_2: float = Field(default=0, description="Cache threshold for transformer_2 (0 to disable caching)")
    video_output_quality: int = Field(default=5, ge=1, le=9, description="Video output quality (1-9)")
    
class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Image-to-Video pipeline and resources here."""
        print("Setting up Wan2.2 Image-to-Video pipeline...")
        # Store resolution defaults
        self.resolution_presets = {
            "480p": {"max_area": 480 * 832},
            "720p": {"max_area": 720 * 1280}
        }
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16
        
        # Get variant and determine if using quantization
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            print(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        
        print(f"Loading model variant: {variant}")
        
        self.model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        
        self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)

        # use default wan image processor to resize and crop the image
        self.image_processor = ModularPipeline.from_pretrained("YiYiXu/WanImageProcessor", trust_remote_code=True)
        
        # Determine if this variant uses offloading and resolution
        use_offloading = variant.endswith("_offload")
        use_cpu_offload = variant.startswith("fp16")
        
        # Extract resolution from variant name
        if variant.endswith("_480p") or variant.endswith("_480p_offload"):
            self.default_resolution = "480p"
        elif variant.endswith("_720p") or variant.endswith("_720p_offload"):
            self.default_resolution = "720p"
        else:
            self.default_resolution = "720p"  # fallback
        
        # Get base variant name for quantized models (remove resolution and offload suffixes)
        base_variant = variant.replace("_offload", "").replace("_480p", "").replace("_720p", "")
        
        if base_variant in ["default", "fp16"]:
            # Load standard F16 pipeline
            print(f"Loading standard F16 Wan2.2 I2V pipeline for {variant}...")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
                vae=self.vae,
                torch_dtype=self.dtype,
            )
        else:
            # Load quantized transformers
            print(f"Loading quantized transformers for {variant}...")
            repo_id = "QuantStack/Wan2.2-I2V-A14B-GGUF"
            variant_files = MODEL_VARIANTS[variant]
            
            high_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['high_noise'])
           
            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )

            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['low_noise'])
            
            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )
            
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                vae=self.vae,
                transformer=transformer_high_noise,  # High noise goes to main transformer
                transformer_2=transformer_low_noise,  # Low noise goes to transformer_2
                torch_dtype=self.dtype,
            )
        
        # Configure UniPCM scheduler with flow_shift=8.0 for faster generation (BEFORE moving to device)
        print("Configuring UniPCM scheduler...")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=8.0)
        
        # Unified offloading strategy BEFORE loading LoRA (following fast-wan.py order)
        if use_cpu_offload:
            # Enable CPU offload for fp16 variants (no device movement needed)
            print("Enabling CPU offload...")
            self.pipe.enable_model_cpu_offload()
        elif use_offloading:
            # Enable group offloading for offload variants
            print("Enabling group offloading...")
            onload_device = self.device
            offload_device = torch.device("cpu")
            
            self.pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            self.pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            #self.pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            #self.pipe.text_encoder.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        
        
        print(f"Moving pipeline to {self.device}...")
        self.pipe.to(self.device)
        
        # Load LightX2V LoRA
        print("Loading LightX2V LoRA...")
        lora_path = hf_hub_download(
            repo_id="Kijai/WanVideo_comfy",
            filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
        )
        
        # Load LoRA weights for both transformers (following fast-wan.py exactly)
        if hasattr(self.pipe, "transformer_2") and self.pipe.transformer_2 is not None:
            # Wan 2.2 - load LoRA for both transformers
            org_state_dict = safetensors.torch.load_file(lora_path)
            converted_state_dict = _convert_non_diffusers_wan_lora_to_diffusers(org_state_dict)
            #self.pipe.load_lora_weights(lora_path, adapter_name='lightx2v_t1')
            #self.pipe.set_adapters(["lightx2v_t1"], adapter_weights=[3.0])
            self.pipe.transformer.load_lora_adapter(converted_state_dict, adapter_name="lightx2v1")
            self.pipe.transformer.set_adapters(["lightx2v1"], weights=[3.0])

            self.pipe.transformer_2.load_lora_adapter(converted_state_dict, adapter_name="lightx2v2")
            self.pipe.transformer_2.set_adapters(["lightx2v2"], weights=[1.5])
        else:
            # Wan 2.1 - load LoRA for single transformer
            self.pipe.load_lora_weights(lora_path)
        
        # Fuse LoRA weights for better performance
        #print("Fusing LoRA weights...")
        #self.pipe.fuse_lora()
        
        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from image and text prompt."""
        print(f"Generating video with prompt: {input_data.prompt}")
        
        # Use resolution preset if max_area not specified
        resolution = input_data.resolution if input_data.resolution is not None else self.default_resolution
        preset = self.resolution_presets.get(resolution, self.resolution_presets["720p"])
        max_area = input_data.max_area if input_data.max_area is not None else preset["max_area"]
        print(f"Using resolution: {resolution}, max area: {max_area}")
        
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