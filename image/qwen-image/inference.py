from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional, Literal
import torch
from huggingface_hub import hf_hub_download
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel, 
    GGUFQuantizationConfig,
    FirstBlockCacheConfig,
    UniPCMultistepScheduler,
)
from diffusers.hooks import apply_group_offloading
import os
from PIL import Image
import logging

# Set up HuggingFace transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from. Supports both English and Chinese text rendering.")
    negative_prompt: Optional[str] = Field(default="", description="The negative prompt to guide what not to include in the image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    num_inference_steps: int = Field(default=50, description="The number of inference steps for generation quality.")
    true_cfg_scale: float = Field(default=4.0, description="The CFG scale for generation guidance.")
    seed: Optional[int] = Field(default=None, description="The seed for reproducible generation.")
    language: Optional[Literal["en", "zh"]] = Field(default="en", description="Language for prompt optimization (English or Chinese).")
    cache_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="First-block cache threshold for transformer (0 disables caching).")
    use_unipcm_flow_matching: bool = Field(default=False, description="If true, switch scheduler to UniPCM flow matching configuration.")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image file.")

# Define available GGUF model variants from city96/Qwen-Image-gguf
GGUF_MODEL_VARIANTS = {
    "q2_k": "qwen-image-Q2_K.gguf",
    "q3_k_s": "qwen-image-Q3_K_S.gguf", 
    "q3_k_m": "qwen-image-Q3_K_M.gguf",
    "q4_0": "qwen-image-Q4_0.gguf",
    "q4_1": "qwen-image-Q4_1.gguf",
    "q4_k_s": "qwen-image-Q4_K_S.gguf",
    "q4_k_m": "qwen-image-Q4_K_M.gguf",
    "q5_0": "qwen-image-Q5_0.gguf",
    "q5_1": "qwen-image-Q5_1.gguf",
    "q5_k_s": "qwen-image-Q5_K_S.gguf",
    "q5_k_m": "qwen-image-Q5_K_M.gguf",
    "q6_k": "qwen-image-Q6_K.gguf",
    "q8_0": "qwen-image-Q8_0.gguf",
    "bf16": "qwen-image-BF16.gguf",
}

DEFAULT_VARIANT = "default"


ASPECT_RATIOS = {}

# Language-specific positive magic prompts for enhanced quality
POSITIVE_MAGIC = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图"
}



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Qwen-Image model and resources."""
        logging.basicConfig(level=logging.INFO)
        
        # Determine which model variant to use
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        low_vram = variant.endswith("_low_vram")
        base_variant = variant[:-9] if low_vram else variant
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logging.info(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        # Initialize the Qwen-Image pipeline
        if base_variant == "default":
            # Use the original Qwen-Image model
            logging.info("Loading original Qwen-Image model from Qwen/Qwen-Image")
            
            self.pipeline = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image",
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            
            # Defer device movement until offloading decision
            
        elif base_variant in GGUF_MODEL_VARIANTS:
            # Use GGUF quantized model from city96
            filename = GGUF_MODEL_VARIANTS[base_variant]
            gguf_repo = "city96/Qwen-Image-gguf"
            original_model_id = "Qwen/Qwen-Image"
            
            logging.info(f"Loading Qwen-Image GGUF variant: {base_variant} ({filename}) from {gguf_repo}")
            
            # Download the GGUF model file
            gguf_path = hf_hub_download(
                repo_id=gguf_repo,
                filename=filename,
                cache_dir="/tmp/qwen_image_cache"
            )
            logging.info(f"GGUF model downloaded to {gguf_path}")
            
            # Load GGUF transformer model similar to flux-1-dev pattern
            logging.info("Loading GGUF quantized transformer...")
                            
            transformer = QwenImageTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.torch_dtype),
                torch_dtype=self.torch_dtype,
                config=original_model_id,
                subfolder="transformer",
            )
            logging.info("Successfully loaded GGUF transformer!")
            
            # Create pipeline with custom GGUF transformer; use default HF text encoder
            self.pipeline = QwenImagePipeline.from_pretrained(
                original_model_id,
                transformer=transformer,
                torch_dtype=self.torch_dtype,
            )
            
            # Defer device movement until offloading decision
                                
        else:
            logging.warning(f"Unknown variant '{variant}', falling back to original model")
            
            self.pipeline = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image",
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            
            # Move to appropriate device and enable optimizations
            self.pipeline = self.pipeline.to(self.device)
        
        # Apply offloading strategy
        if low_vram:
            logging.info("Enabling leaf-level group offloading (low_vram mode)")
            onload_device = self.device
            offload_device = torch.device("cpu")

            self.pipeline.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            self.pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(self.pipeline.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")

            logging.info("Group offloading configured; skipping explicit .to(device)")
        else:
            # Default: enable model CPU offload
            logging.info("Enabling model CPU offload (default mode)")
            self.pipeline.enable_model_cpu_offload()

        # Common optimizations
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()

        logging.info("Qwen-Image pipeline initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate image using Qwen-Image pipeline."""
        logging.info("Running in text-to-image generation mode")
        
        # Optionally switch to UniPC flow matching scheduler
        if input_data.use_unipcm_flow_matching:
            logging.info("Switching scheduler to UniPCM flow matching")
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
            )
        
        # Use provided width/height directly
        width, height = input_data.width, input_data.height
        logging.info(f"Using dimensions: {width}x{height}")
        
        # Enhance prompt with language-specific magic
        enhanced_prompt = input_data.prompt
        if input_data.language in POSITIVE_MAGIC:
            enhanced_prompt = f"{input_data.prompt} {POSITIVE_MAGIC[input_data.language]}"
            logging.info(f"Enhanced prompt with {input_data.language} magic")
        
        # Set up generator for reproducibility
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
            logging.info(f"Using seed: {input_data.seed}")

        # Configure first-block caching if requested
        # Disable any existing caches to avoid conflicts
        if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'disable_cache'):
            self.pipeline.transformer.disable_cache()
        if input_data.cache_threshold and input_data.cache_threshold > 0:
            logging.info(f"Enabling first-block cache with threshold={input_data.cache_threshold}")
            cache_cfg = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipeline.transformer.enable_cache(cache_cfg)
        
        # Generate the image
        logging.info(f"Generating image for prompt: '{input_data.prompt[:100]}...'")
        
        result = self.pipeline(
            prompt=enhanced_prompt,
            negative_prompt=input_data.negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=input_data.num_inference_steps,
            true_cfg_scale=input_data.true_cfg_scale,
            generator=generator
        )
        
        # Extract the generated image
        if hasattr(result, 'images') and result.images:
            image = result.images[0]
        elif isinstance(result, list) and len(result) > 0:
            image = result[0]
        else:
            raise RuntimeError("No image generated from pipeline")
        
        # Save the image
        output_path = "/tmp/qwen_generated_image.png"
        image.save(output_path, format="PNG")
        
        logging.info(f"Image generation completed and saved to {output_path}")
        
        return AppOutput(image_output=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logging.info("Resources cleaned up")