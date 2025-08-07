from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional, Literal
import torch
from huggingface_hub import hf_hub_download
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel, 
    GGUFQuantizationConfig
)
import os
from PIL import Image
import logging

# Set up HuggingFace transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from. Supports both English and Chinese text rendering.")
    negative_prompt: Optional[str] = Field(default="", description="The negative prompt to guide what not to include in the image.")
    width: int = Field(default=1328, description="The width in pixels of the generated image.")
    height: int = Field(default=1328, description="The height in pixels of the generated image.")
    aspect_ratio: Optional[Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]] = Field(default="1:1", description="Predefined aspect ratio that overrides width/height if specified.")
    num_inference_steps: int = Field(default=50, description="The number of inference steps for generation quality.")
    true_cfg_scale: float = Field(default=4.0, description="The CFG scale for generation guidance.")
    seed: Optional[int] = Field(default=None, description="The seed for reproducible generation.")
    language: Optional[Literal["en", "zh"]] = Field(default="en", description="Language for prompt optimization (English or Chinese).")

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

# Aspect ratio definitions matching Qwen-Image documentation
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

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
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logging.info(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        # Initialize the Qwen-Image pipeline
        try:
            if variant == "default":
                # Use the original Qwen-Image model
                logging.info("Loading original Qwen-Image model from Qwen/Qwen-Image")
                
                self.pipeline = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image",
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True
                )
                
                # Move to appropriate device and enable optimizations
                self.pipeline = self.pipeline.to(self.device)
                
            elif variant in GGUF_MODEL_VARIANTS:
                # Use GGUF quantized model from city96
                filename = GGUF_MODEL_VARIANTS[variant]
                gguf_repo = "city96/Qwen-Image-gguf"
                original_model_id = "Qwen/Qwen-Image"
                
                logging.info(f"Loading Qwen-Image GGUF variant: {variant} ({filename}) from {gguf_repo}")
                
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
                )
                logging.info("Successfully loaded GGUF transformer!")
                
                # Create pipeline with custom GGUF transformer
                self.pipeline = QwenImagePipeline.from_pretrained(
                    original_model_id,
                    transformer=transformer,
                    torch_dtype=self.torch_dtype,
                )
                
                # Move to appropriate device
                self.pipeline = self.pipeline.to(self.device)
                                    
            else:
                logging.warning(f"Unknown variant '{variant}', falling back to original model")
                
                self.pipeline = QwenImagePipeline.from_pretrained(
                    "Qwen/Qwen-Image",
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True
                )
                
                # Move to appropriate device and enable optimizations
                self.pipeline = self.pipeline.to(self.device)
            
            # Enable common optimizations if available
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
                
            logging.info("Qwen-Image pipeline initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Qwen-Image pipeline: {e}")
            raise

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate image using Qwen-Image pipeline."""
        try:
            logging.info("Running in text-to-image generation mode")
            
            # Determine dimensions from aspect ratio or use provided width/height
            if input_data.aspect_ratio and input_data.aspect_ratio in ASPECT_RATIOS:
                width, height = ASPECT_RATIOS[input_data.aspect_ratio]
                logging.info(f"Using aspect ratio {input_data.aspect_ratio}: {width}x{height}")
            else:
                width, height = input_data.width, input_data.height
                logging.info(f"Using custom dimensions: {width}x{height}")
            
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
            
        except Exception as e:
            logging.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logging.info("Resources cleaned up")