import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field, BaseModel
from diffusers import AutoencoderKLWan, WanTransformer3DModel, GGUFQuantizationConfig, WanPipeline
from diffusers.hooks import FirstBlockCacheConfig
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from PIL import Image
import logging
import requests
import re
from datetime import datetime

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_url: str = Field(description="URL to LoRA file (.safetensors) or Civitai model page")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the LoRA effect")
    target_transformer: str = Field(default="both", description="Target transformer: 'high', 'low', or 'both'")

def get_civit_download_url(model_id):
    civitai_token = os.environ.get('CIVITAI_TOKEN')
    if civitai_token:
        url = f"https://civitai.com/models/{model_id}&token={civitai_token}"
    else:
        url = f"https://civitai.com/models/{model_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        versions = data["modelVersions"]
        def get_date(item):
            date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
            try:
                # Handle ISO format dates (most common format from APIs)
                if 'T' in date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(date_str)
                return dt
            except Exception as e:
                return datetime(1970, 1, 1)
        sorted_data = sorted(versions, key=get_date, reverse=True)
        # Filter for Wan/video models if available, otherwise use latest
        wan_versions = [v for v in sorted_data if v.get('baseModel', '').lower().startswith('wan')]
        latest_version = wan_versions[0] if wan_versions else sorted_data[0]
        downloadUrl = latest_version["downloadUrl"]
        return downloadUrl
    except Exception as e:
        logging.error(f"Failed to fetch Civitai model info: {e}")
        return None

def download_model_data(model_id, downloadUrl, targetFolder):
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    filePath = os.path.join(targetFolder, f"{model_id}.safetensors")
    if not os.path.isfile(filePath):
        # If downloading from civitai, add token if present
        civitai_token = os.environ.get('CIVITAI_TOKEN')
        if civitai_token:
            print("Using CivitAI token from environment variable CIVITAI_TOKEN")
        else:
            print("WARNING: No CivitAI token found in environment variable CIVITAI_TOKEN, set one if you get a 403 error")

        if 'civitai.com' in downloadUrl and civitai_token:
            if '?' in downloadUrl:
                downloadUrl = f"{downloadUrl}&token={civitai_token}"
            else:
                downloadUrl = f"{downloadUrl}?token={civitai_token}"
        response = requests.get(downloadUrl)
        with open(filePath, 'wb') as file:
            file.write(response.content)
    return filePath

def load_lora_adapter(pipeline, lora_url, adapter_name="lora", lora_multiplier=1.0, target_transformer="both"):
    """
    Load a LoRA adapter using the modern Diffusers PEFT integration.
    Supports Hugging Face repos, direct URLs, and Civitai models.
    
    Args:
        pipeline: The WanPipeline instance
        lora_url: URL to the LoRA file
        adapter_name: Name for the adapter
        lora_multiplier: Scale for the LoRA effect
        target_transformer: Which transformer to load on ('high', 'low', or 'both')
    """
    if not lora_url:
        return False
    
    def _load_lora_with_target(load_kwargs):
        """Helper function to load LoRA on specified transformer(s)"""
        if target_transformer == "both":
            # Load on both transformers (original behavior)
            pipeline.load_lora_weights(**load_kwargs)
        elif target_transformer == "high":
            # Load only on main transformer (high noise)
            load_kwargs["load_into_transformer"] = True
            load_kwargs["load_into_transformer_2"] = False
            pipeline.load_lora_weights(**load_kwargs)
        elif target_transformer == "low":
            # Load only on transformer_2 (low noise)
            load_kwargs["load_into_transformer"] = False
            load_kwargs["load_into_transformer_2"] = True
            pipeline.load_lora_weights(**load_kwargs)
        else:
            logging.error(f"Invalid target_transformer: {target_transformer}. Must be 'high', 'low', or 'both'")
            return False
        return True
    
    try:
        # 1. Hugging Face blob URL (e.g., https://huggingface.co/username/repo/blob/main/file.safetensors)
        if "huggingface.co" in lora_url and "/blob/" in lora_url:
            # Convert blob URL to repo format
            parts = lora_url.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'blob' in parts:
                repo_start = parts.index('huggingface.co') + 1
                blob_index = parts.index('blob')
                repo_id = '/'.join(parts[repo_start:blob_index])
                weight_name = '/'.join(parts[blob_index + 2:])  # Skip 'blob' and branch name
                load_kwargs = {"repo_id": repo_id, "weight_name": weight_name, "adapter_name": adapter_name, "lora_scale": lora_multiplier}
                if _load_lora_with_target(load_kwargs):
                    logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name} on {target_transformer}")
                    return True
                return False
            else:
                logging.error(f"Invalid Hugging Face blob URL format: {lora_url}")
                return False
        
        # 2. Hugging Face resolve URL (e.g., https://huggingface.co/username/repo/resolve/main/file.safetensors)
        elif "huggingface.co" in lora_url and "/resolve/" in lora_url:
            # Convert resolve URL to repo format
            parts = lora_url.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'resolve' in parts:
                repo_start = parts.index('huggingface.co') + 1
                resolve_index = parts.index('resolve')
                repo_id = '/'.join(parts[repo_start:resolve_index])
                weight_name = '/'.join(parts[resolve_index + 2:])  # Skip 'resolve' and branch name
                load_kwargs = {"repo_id": repo_id, "weight_name": weight_name, "adapter_name": adapter_name, "lora_scale": lora_multiplier}
                if _load_lora_with_target(load_kwargs):
                    logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name} on {target_transformer}")
                    return True
                return False
            else:
                logging.error(f"Invalid Hugging Face resolve URL format: {lora_url}")
                return False
        
        # 3. Hugging Face repository string (e.g., "username/repo" or "username/repo/filename.safetensors")
        elif "/" in lora_url and not lora_url.startswith('http') and "civitai.com" not in lora_url:
            parts = lora_url.split('/')
            if len(parts) == 2:
                # Format: "username/repo" - let Diffusers auto-detect the weight file
                repo_id = lora_url
                load_kwargs = {"repo_id": repo_id, "adapter_name": adapter_name, "lora_scale": lora_multiplier}
                if _load_lora_with_target(load_kwargs):
                    logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id} on {target_transformer}")
                    return True
                return False
            elif len(parts) > 2:
                # Format: "username/repo/filename.safetensors"
                repo_id = '/'.join(parts[:2])
                weight_name = '/'.join(parts[2:])
                load_kwargs = {"repo_id": repo_id, "weight_name": weight_name, "adapter_name": adapter_name, "lora_scale": lora_multiplier}
                if _load_lora_with_target(load_kwargs):
                    logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name} on {target_transformer}")
                    return True
                return False
        
        # 4. Other direct .safetensors URL
        elif lora_url.endswith('.safetensors') and lora_url.startswith('http'):
            # Download to local cache
            lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
            if not os.path.exists(lora_dir):
                os.makedirs(lora_dir)
            
            model_id = os.path.splitext(os.path.basename(lora_url))[0]
            lora_path = download_model_data(model_id, lora_url, lora_dir)
            load_kwargs = {"pretrained_model_name_or_path": lora_path, "adapter_name": adapter_name, "lora_scale": lora_multiplier}
            if _load_lora_with_target(load_kwargs):
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from URL: {lora_url} on {target_transformer}")
                return True
            return False
        
        # 5. Civitai model URL
        elif "civitai.com" in lora_url:
            match = re.search(r"/models/(\d+)", lora_url)
            if not match:
                logging.error("Could not extract model ID from Civitai URL")
                return False
            
            model_id = match.group(1)
            download_url = get_civit_download_url(model_id)
            if not download_url:
                logging.error(f"No download URL found for Civitai model {model_id}")
                return False
            
            lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
            if not os.path.exists(lora_dir):
                os.makedirs(lora_dir)
            
            lora_path = download_model_data(model_id, download_url, lora_dir)
            load_kwargs = {"pretrained_model_name_or_path": lora_path, "adapter_name": adapter_name, "lora_scale": lora_multiplier}
            if _load_lora_with_target(load_kwargs):
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from Civitai model: {model_id} on {target_transformer}")
                return True
            return False
        
        else:
            logging.warning(f"Unsupported LoRA URL format: {lora_url}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to load LoRA adapter '{adapter_name}': {e}")
        return False

# Model variants mapping for GGUF quantization from QuantStack
# Includes both HighNoise and LowNoise transformers for better quality
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
    prompt: str = Field(description="Text prompt for image generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    width: Optional[int] = Field(default=1024, description="Width of the generated image")
    height: Optional[int] = Field(default=1024, description="Height of the generated image")
    guidance_scale: float = Field(default=3.5, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=40, description="Number of denoising steps")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0, description="Cache threshold for transformer (0 to disable caching)")
    cache_threshold_2: float = Field(default=0, description="Cache threshold for transformer_2 (0 to disable caching)")
    boundary_ratio: float = Field(default=0.875, description="Boundary ratio for dual transformer setup (0.0-1.0, higher values give more steps to high noise transformer)")
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")
    
class AppOutput(BaseAppOutput):
    image: File = Field(description="output image")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Text-to-Image pipeline and resources here."""
        print("Setting up Wan2.2 Text-to-Image pipeline...")
        
        # Initialize LoRA tracking attributes
        self.loaded_loras = {}  # adapter_name -> (lora_url, lora_multiplier)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
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
            "Wan-AI/Wan2.1-T2V-14B-Diffusers", 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        if variant == "default":
            # Load standard F16 pipeline with both high and low noise transformers
            print("Loading standard F16 Wan2.2 T2I pipeline with dual transformers...")
            
            # Load high noise transformer
            transformer_high_noise = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )

            # Load low noise transformer
            transformer_low_noise = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )

            self.pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                vae=self.vae,
                transformer=transformer_high_noise,  # High noise goes to main transformer
                transformer_2=transformer_low_noise,  # Low noise goes to transformer_2
                boundary_ratio=0.875,  # Default boundary ratio, will be updated per-request
                torch_dtype=self.dtype
            )

            self.pipe.enable_model_cpu_offload()
        else:
            # Load quantized transformers
            print(f"Loading quantized transformers for {variant}...")
            repo_id = "QuantStack/Wan2.2-T2V-A14B-GGUF"
            variant_files = MODEL_VARIANTS[variant]
            
            # Download both high and low noise models
            high_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['high_noise'])
            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['low_noise'])
           
            # Load high noise transformer
            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )

            # Load low noise transformer
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
                boundary_ratio=0.875,  # Default boundary ratio, will be updated per-request
                torch_dtype=self.dtype
            )
            
            self.pipe.enable_model_cpu_offload()


        
        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate image from text prompt."""
        print(f"Generating image with prompt: {input_data.prompt}")
        
        # Configure caching if thresholds are non-zero
        # First disable any existing caching to prevent conflicts
        if hasattr(self.pipe.transformer, 'disable_cache'):
            self.pipe.transformer.disable_cache()
        if hasattr(self.pipe, 'transformer_2') and hasattr(self.pipe.transformer_2, 'disable_cache'):
            self.pipe.transformer_2.disable_cache()
        
        if input_data.cache_threshold > 0:
            print(f"Enabling cache for transformer with threshold: {input_data.cache_threshold}")
            cache_config = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipe.transformer.enable_cache(cache_config)
            
        if input_data.cache_threshold_2 > 0 and hasattr(self.pipe, 'transformer_2'):
            print(f"Enabling cache for transformer_2 with threshold: {input_data.cache_threshold_2}")
            cache_config_2 = FirstBlockCacheConfig(threshold=input_data.cache_threshold_2)
            self.pipe.transformer_2.enable_cache(cache_config_2)
        
        # Handle LoRA adapters
        loras = getattr(input_data, "loras", None) or []
        current_adapters = set(self.loaded_loras.keys())
        requested_adapters = set(lora.adapter_name for lora in loras)

        # Unload adapters that are no longer requested or changed
        for adapter_name in list(current_adapters):
            found = next((l for l in loras if l.adapter_name == adapter_name), None)
            if not found or (
                self.loaded_loras[adapter_name][0] != found.lora_url or
                self.loaded_loras[adapter_name][1] != found.lora_multiplier or
                self.loaded_loras[adapter_name][2] != found.target_transformer
            ):
                try:
                    self.pipe.delete_adapters(adapter_name)
                    logging.info(f"Unloaded previous LoRA adapter: {adapter_name}")
                except Exception as e:
                    logging.warning(f"Failed to unload previous LoRA adapter {adapter_name}: {e}")
                del self.loaded_loras[adapter_name]

        # Load and activate requested adapters
        adapter_weights = []
        active_adapters = []
        for lora in loras:
            if (
                lora.adapter_name not in self.loaded_loras or
                self.loaded_loras[lora.adapter_name][0] != lora.lora_url or
                self.loaded_loras[lora.adapter_name][1] != lora.lora_multiplier or
                self.loaded_loras[lora.adapter_name][2] != lora.target_transformer
            ):
                success = load_lora_adapter(self.pipe, lora.lora_url, lora.adapter_name, lora.lora_multiplier, lora.target_transformer)
                if success:
                    self.loaded_loras[lora.adapter_name] = (lora.lora_url, lora.lora_multiplier, lora.target_transformer)
            if lora.adapter_name in self.loaded_loras:
                active_adapters.append(lora.adapter_name)
                adapter_weights.append(lora.lora_multiplier)

        # Set all active adapters
        if active_adapters:
            self.pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)
            logging.info(f"Activated LoRA adapters: {active_adapters} with weights: {adapter_weights}")
        
        # Use resolution preset if width/height not specified
        width = input_data.width
        height = input_data.height
                
        # Set seed if provided
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(input_data.seed)
        
        
        # Generate image
        print("Starting image generation...")
        output = self.pipe(
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            height=height,
            width=width,
            num_frames=1,  # Required for text-to-image to create proper temporal dimension
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
        ).frames[0]
        
        print("Generation complete, exporting...")
        
        # Create temporary file for image output
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Process the output - output is a list of frames (with num_frames=1)
        if isinstance(output, list) and len(output) > 0:
            frame = output[0]  # Get the first frame
        else:
            frame = output
        
        # Handle PIL Images directly
        if hasattr(frame, 'save'):  # PIL Image
            frame.save(output_path)
            print(f"Image exported to: {output_path}")
            return AppOutput(image=File(path=output_path))
        
        # Convert tensor to numpy array if needed
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        # Handle different frame shapes
        print(f"Frame shape: {frame.shape}")
        
        # Remove single dimensions and ensure proper format
        frame = np.squeeze(frame)  # Remove dimensions of size 1
        
        # Ensure we have the right shape (H, W, C) or (H, W)
        if len(frame.shape) == 3:
            if frame.shape[0] == 3:  # CHW format
                frame = frame.transpose(1, 2, 0)  # Convert to HWC
        elif len(frame.shape) == 4:  # Batch dimension
            frame = frame[0]  # Take first item in batch
            if frame.shape[0] == 3:  # CHW format
                frame = frame.transpose(1, 2, 0)  # Convert to HWC
        
        # Ensure values are in [0, 1] range, then convert to [0, 255] uint8
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        
        img = Image.fromarray(frame)
        img.save(output_path)
        
        print(f"Image exported to: {output_path}")
        return AppOutput(image=File(path=output_path))

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