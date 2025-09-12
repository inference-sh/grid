import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "YES"

import torch
import gc
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field
from PIL import Image
from diffusers import WanImageToVideoPipeline, GGUFQuantizationConfig, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

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


class LocalWan14BImageProcessor:
    """
    Minimal local replacement for YiYiXu/WanImageProcessor block.
    - Resizes input image to fit within max_area while preserving aspect ratio
    - Ensures height/width are multiples of (vae_stride * patch_size) per dimension
    """

    def __init__(self, patch_size=(1, 2, 2), vae_stride=(4, 8, 8)) -> None:
        self.patch_size = patch_size
        self.vae_stride = vae_stride

    def __call__(self, image, max_area=None, output="processed_image") -> Image.Image:
        # Accept path or PIL Image
        if isinstance(image, (str, os.PathLike)):
            pil = load_image(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image
        else:
            raise ValueError(f"Invalid image type: {type(image)}; only support PIL Image or path string")

        if max_area is None:
            max_area = 480 * 832

        aspect_ratio = pil.height / pil.width
        mod_value_height = self.vae_stride[1] * self.patch_size[1]
        mod_value_width = self.vae_stride[2] * self.patch_size[2]

        import numpy as np
        height = int(round(np.sqrt(max_area * aspect_ratio))) // mod_value_height * mod_value_height
        width = int(round(np.sqrt(max_area / aspect_ratio))) // mod_value_width * mod_value_width
        if height < mod_value_height:
            height = mod_value_height
        if width < mod_value_width:
            width = mod_value_width

        resized = pil.resize((width, height))

        print(f" initial image size: {pil.size}")
        print(f" processed image size: {resized.size}")

        return resized

class AppInput(BaseAppInput):
    image: File = Field(description="Input image for video generation")
    end_image: Optional[File] = Field(default=None, description="Optional end frame image; when provided, the video will transition from the first frame to this last frame")
    prompt: str = Field(description="Text prompt for video generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    num_frames: int = Field(default=81, description="Number of frames to generate")
    num_inference_steps: int = Field(default=4, description="Number of denoising steps (4 or 6 are typically enough for Lightning)")
    fps: int = Field(default=16, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    boundary_ratio: float = Field(default=0.875, ge=0.0, le=1.0, description="Boundary ratio between high and low noise transformers (0-1)")
    
class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Image-to-Video pipeline and resources here."""
        print("Setting up Wan2.2 Image-to-Video pipeline (Lightning LoRA)...")
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
        
        self.model_id = "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers"
        
        self.vae = AutoencoderKLWan.from_pretrained('magespace/Wan2.1-I2V-14B-480P-Lightning-Diffusers', subfolder="vae", torch_dtype=torch.float32)

        # Use local WAN image processor (replacement for YiYiXu/WanImageProcessor ModularPipeline)
        self.image_processor = LocalWan14BImageProcessor()
        
        # Determine offloading strategy from variant suffix
        # New convention: *_offload -> model CPU offload, *_offload_lowvram -> leaf/group offload
        use_cpu_offload = (variant.endswith("_offload") or variant.endswith("_cpu_offload") or variant.startswith("fp16"))
        use_offloading = variant.endswith("_offload_lowvram") and not use_cpu_offload
        
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
                "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers", 
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
                config="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )

            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['low_noise'])
            
            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )
            
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
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
            self.pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            #self.pipe.text_encoder.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        
        
        # Respect offloading: do not move pipeline explicitly when offloading is enabled
        if use_cpu_offload or use_offloading:
            print("Offloading enabled; not moving pipeline explicitly.")
        else:
            print(f"Moving pipeline to {self.device}...")
            self.pipe.to(self.device)
        
        # Load Lightning LoRAs without fusing
        #print("Loading Lightning LoRAs...")

        #self.pipe.load_lora_weights(
        #    "Kijai/WanVideo_comfy",
        #    weight_name="Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors",
        #    adapter_name="lightning",
        #)

        #kwargs = {}
        #kwargs["load_into_transformer_2"] = True
        #self.pipe.load_lora_weights(
        #    "Kijai/WanVideo_comfy",
        #    weight_name="Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors",
        #    adapter_name="lightning_2",
        #    **kwargs,
        #)

        #self.pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.25, 1.25])
        
        # Set flash attention backend
        self.pipe.transformer.set_attention_backend("_flash_3_hub")
        
        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from image and text prompt."""
        print(f"Generating video with prompt: {input_data.prompt}")
        
        # Use resolution preset based on selected variant
        resolution = self.default_resolution
        preset = self.resolution_presets.get(resolution, self.resolution_presets["720p"])
        max_area = preset["max_area"]
        print(f"Using resolution: {resolution}, max area: {max_area}")
        
        # Load and process input image (ensure file handle is closed)
        with Image.open(input_data.image.path) as pil_image:
            pil_image = pil_image.convert("RGB")
            print(f"Loaded image: {pil_image.size}")
        
        # Resize image according to pipeline requirements
        resized_image = self.image_processor(
            image=input_data.image.path,
            max_area=max_area, output="processed_image")
        
        width = resized_image.width
        height = resized_image.height

        # Optionally process end image for last frame transition
        last_image = None
        if input_data.end_image is not None:
            resized_end_image = self.image_processor(
                image=input_data.end_image.path,
                max_area=max_area,
                output="processed_image",
            )
            if resized_end_image.width != width or resized_end_image.height != height:
                try:
                    resized_end_image = resized_end_image.resize((width, height), Image.Resampling.BILINEAR)
                except Exception:
                    resized_end_image = resized_end_image.resize((width, height))
            last_image = resized_end_image
        
        # Set seed if provided
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
            print(f"Using seed: {input_data.seed}")
            
        # Generate video
        print("Starting video generation...")
        # Update boundary ratio at runtime
        print(f"Updating boundary ratio to: {input_data.boundary_ratio}")
        self.pipe.register_to_config(boundary_ratio=input_data.boundary_ratio)
        
        with torch.inference_mode():
            output = self.pipe(
                image=resized_image,
                last_image=last_image,
                prompt=input_data.prompt,
                negative_prompt=input_data.negative_prompt,
                height=height,
                width=width,
                num_frames=input_data.num_frames,
                guidance_scale=1.0,
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

        # Cleanup large objects and GPU cache to prevent memory growth across runs
        try:
            del output
            del resized_image
            del generator
        except Exception:
            pass
        finally:
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

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

