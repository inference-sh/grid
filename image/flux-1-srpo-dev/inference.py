from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from diffusers import (
    FluxPipeline, 
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    FlowMatchEulerDiscreteScheduler
)
from transformers import T5EncoderModel
import os
from PIL import Image
import logging
import json
import requests
from datetime import datetime
from accelerate import Accelerator
from dateutil.parser import parse as parse_date
import urllib.parse
import re
from enum import Enum

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_url: str = Field(description="URL to LoRA file (.safetensors) or Civitai model page")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the LoRA effect")

# Define enums for sampler and scheduler
class SchedulerEnum(str, Enum):
    normal = "normal"
    karras = "karras"
    exponential = "exponential"
    beta = "beta"

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    num_inference_steps: int = Field(default=30, description="The number of inference steps.")
    guidance_scale: float = Field(default=3.5, description="The guidance scale.")
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")
    scheduler: SchedulerEnum = Field(default=SchedulerEnum.normal, description="Scheduler to use for diffusion process.")
    denoise: float = Field(default=1.0, ge=0.0, le=1.0, description="Denoising strength (0.0 to 1.0, where 1.0 is standard full denoising)")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image.")

MODEL_VARIANTS = {
    "default": "srpo-BF16.gguf",
    "q2_k": "srpo-Q2_K.gguf",
    "q3_k": "srpo-Q3_K.gguf",
    "q4_k": "srpo-Q4_K.gguf",
    "q5_k": "srpo-Q5_K.gguf",
    "q6_k": "srpo-Q6_K.gguf",
    "q8_0": "srpo-Q8_0.gguf",
}
DEFAULT_VARIANT = "default"

def get_civit_download_url(model_id):
    url = f"https://civitai.com/models/{model_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        versions = data["modelVersions"]
        def get_date(item):
            date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
            try:
                dt = parse_date(date_str)
                return dt
            except Exception as e:
                return parse_date('1970-01-01T00:00:00Z')
        sorted_data = sorted(versions, key=get_date, reverse=True)
        versions = [v for v in sorted_data if v.get('baseModel') == 'Flux.1 D']

        latest_version = versions[0]
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
        if 'civitai.com' in downloadUrl and civitai_token:
            if '?' in downloadUrl:
                downloadUrl = f"{downloadUrl}&token={civitai_token}"
            else:
                downloadUrl = f"{downloadUrl}?token={civitai_token}"
        response = requests.get(downloadUrl)
        with open(filePath, 'wb') as file:
            file.write(response.content)
    return filePath

def load_lora_adapter(pipeline, lora_url, adapter_name="lora", lora_multiplier=1.0):
    """
    Load a LoRA adapter using the modern Diffusers PEFT integration.
    Supports Hugging Face repos, direct URLs, and Civitai models.
    """
    if not lora_url:
        return False
    
    try:
        # 1. Hugging Face blob URL (e.g., https://huggingface.co/username/repo/blob/main/file.safetensors)
        if "huggingface.co" in lora_url and "/blob/" in lora_url:
            # Convert blob URL to repo format
            # Example: https://huggingface.co/XLabs-AI/flux-lora-collection/blob/main/anime_lora.safetensors
            # -> repo_id: "XLabs-AI/flux-lora-collection", weight_name: "anime_lora.safetensors"
            parts = lora_url.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'blob' in parts:
                repo_start = parts.index('huggingface.co') + 1
                blob_index = parts.index('blob')
                repo_id = '/'.join(parts[repo_start:blob_index])
                weight_name = '/'.join(parts[blob_index + 2:])  # Skip 'blob' and branch name
                pipeline.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name, lora_scale=lora_multiplier)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name}")
                return True
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
                pipeline.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name, lora_scale=lora_multiplier)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name}")
                return True
            else:
                logging.error(f"Invalid Hugging Face resolve URL format: {lora_url}")
                return False
        
        # 3. Hugging Face repository string (e.g., "username/repo" or "username/repo/filename.safetensors")
        elif "/" in lora_url and not lora_url.startswith('http') and "civitai.com" not in lora_url:
            parts = lora_url.split('/')
            if len(parts) == 2:
                # Format: "username/repo" - let Diffusers auto-detect the weight file
                repo_id = lora_url
                pipeline.load_lora_weights(repo_id, adapter_name=adapter_name, lora_scale=lora_multiplier)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}")
                return True
            elif len(parts) > 2:
                # Format: "username/repo/filename.safetensors"
                repo_id = '/'.join(parts[:2])
                weight_name = '/'.join(parts[2:])
                pipeline.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name, lora_scale=lora_multiplier)
                logging.info(f"Loaded LoRA adapter '{adapter_name}' from HF repo: {repo_id}, file: {weight_name}")
                return True
        
        # 4. Other direct .safetensors URL
        elif lora_url.endswith('.safetensors') and lora_url.startswith('http'):
            # Download to local cache
            lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
            if not os.path.exists(lora_dir):
                os.makedirs(lora_dir)
            
            model_id = os.path.splitext(os.path.basename(lora_url))[0]
            lora_path = download_model_data(model_id, lora_url, lora_dir)
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name, lora_scale=lora_multiplier)
            logging.info(f"Loaded LoRA adapter '{adapter_name}' from URL: {lora_url}")
            return True
        
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
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name, lora_scale=lora_multiplier)
            logging.info(f"Loaded LoRA adapter '{adapter_name}' from Civitai model: {model_id}")
            return True
        
        else:
            logging.warning(f"Unsupported LoRA URL format: {lora_url}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to load LoRA adapter '{adapter_name}': {e}")
        return False

def resolve_and_load_lora(pipeline, lora_url, lora_multiplier=1.0):
    """
    Legacy function for backward compatibility.
    Use load_lora_adapter for new implementations.
    """
    return load_lora_adapter(pipeline, lora_url, "lora", lora_multiplier)

class App(BaseApp):
    async def setup(self, metadata):
        # Initialize LoRA tracking attributes
        self.loaded_loras = {}  # adapter_name -> (lora_url, lora_multiplier)
        self.last_lora_url = None
        self.last_lora_multiplier = None
        self.last_lora_adapter_name = None
        
        logging.basicConfig(level=logging.INFO)
        repo_id = "befox/SRPO-GGUF"
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        filename = MODEL_VARIANTS[variant]
        self.original_model_id = "black-forest-labs/FLUX.1-dev"
        
        # if variant == "default":
        #     # Load the BF16 GGUF version
        #     ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        #     logging.info(f"Model downloaded to {ckpt_path}")
            
        #     # Create pipeline with GGUF model
        #     self.pipeline = FluxPipeline.from_pretrained(
        #         "black-forest-labs/FLUX.1-dev",
        #         torch_dtype=torch.bfloat16,
        #     )
        # else:
        logging.info(f"Downloading {filename} from {repo_id}...")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logging.info(f"Model downloaded to {ckpt_path}")
        
        # Load custom text encoder
        logging.info("Loading T5 text encoder...")
        
        #text_encoder = T5EncoderModel.from_pretrained(
        #    "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
        #    gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
        #    torch_dtype=torch.bfloat16
        #)

        text_encoder_2 = T5EncoderModel.from_pretrained(self.original_model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

        # Load quantized transformer
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=self.original_model_id,
            torch_dtype=torch.bfloat16,
            subfolder="transformer"
        )
        
        # Create pipeline with custom components
        
        self.pipeline = FluxPipeline.from_pretrained(
            self.original_model_id,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()

    def manage_lora_adapters(self, loras):
        """Manage LoRA adapters: unload unused ones and load/activate requested ones.
        Returns tuple of (active_adapters, adapter_weights)."""
        current_adapters = set(self.loaded_loras.keys())
        requested_adapters = {lora.adapter_name for lora in loras}
        
        # Unload unused adapters
        for adapter_name in current_adapters - requested_adapters:
            try:
                self.pipeline.delete_adapters(adapter_name)
                logging.info(f"Unloaded LoRA adapter: {adapter_name}")
                del self.loaded_loras[adapter_name]
            except Exception:
                logging.warning(f"Failed to unload LoRA adapter {adapter_name}")
        
        # Load and activate requested adapters
        active_adapters, adapter_weights = [], []
        for lora in loras:
            needs_loading = (
                lora.adapter_name not in self.loaded_loras or
                self.loaded_loras[lora.adapter_name] != (lora.lora_url, lora.lora_multiplier)
            )
            
            if needs_loading and load_lora_adapter(self.pipeline, lora.lora_url, lora.adapter_name, lora.lora_multiplier):
                self.loaded_loras[lora.adapter_name] = (lora.lora_url, lora.lora_multiplier)
            
            if lora.adapter_name in self.loaded_loras:
                active_adapters.append(lora.adapter_name)
                adapter_weights.append(lora.lora_multiplier)
        
        return active_adapters, adapter_weights

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Handle LoRA adapters
        loras = getattr(input_data, "loras", None) or []
        active_adapters, adapter_weights = self.manage_lora_adapters(loras)

        # Set all active adapters
        if active_adapters:
            self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)

        # Configure scheduler based on input using FlowMatchEulerDiscreteScheduler parameters
        scheduler_config = dict(self.pipeline.scheduler.config)
        
        # Handle None scheduler value by defaulting to normal
        selected_scheduler = input_data.scheduler if input_data.scheduler is not None else SchedulerEnum.normal
        
        # Reset all sigma types to False first
        scheduler_config.update({
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False
        })
        
        # Apply scheduler-specific configuration
        if selected_scheduler == SchedulerEnum.karras:
            scheduler_config["use_karras_sigmas"] = True
            logging.info("Using Karras sigmas")
        elif selected_scheduler == SchedulerEnum.exponential:
            scheduler_config["use_exponential_sigmas"] = True
            logging.info("Using exponential sigmas")
        elif selected_scheduler == SchedulerEnum.beta:
            scheduler_config["use_beta_sigmas"] = True
            logging.info("Using beta sigmas")
        elif selected_scheduler == SchedulerEnum.normal:
            # Normal/default configuration
            logging.info("Using normal (default) scheduler configuration")
        
        # Create new scheduler with updated config
        new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self.pipeline.scheduler = new_scheduler
        
        logging.info(f"Scheduler: {selected_scheduler}, Denoise: {input_data.denoise}")
        
        prompt = input_data.prompt
        height = input_data.height
        width = input_data.width
        num_inference_steps = input_data.num_inference_steps
        guidance_scale = input_data.guidance_scale
        seed = input_data.seed
        generator = None
        
        # Use accelerator.device for generator device
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        logging.info(f"Generating image for prompt: '{prompt}'")
        
        # Note: FluxPipeline doesn't support denoise parameter directly
        # The denoise value could be used to modify num_inference_steps if needed
        effective_steps = int(num_inference_steps * input_data.denoise) if input_data.denoise < 1.0 else num_inference_steps
        
        image = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=effective_steps, 
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/generated_image.png"
        image.save(output_path)
        return AppOutput(image_output=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        self.pipeline = None