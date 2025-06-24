from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from diffusers import (
    FluxPipeline, 
    FluxTransformer2DModel,
    GGUFQuantizationConfig
)
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    num_inference_steps: int = Field(default=30, description="The number of inference steps.")
    guidance_scale: float = Field(default=3.5, description="The guidance scale.")
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")
    lora_url: Optional[str] = Field(default=None, description="URL to LoRA file (.safetensors) or Civitai model page")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the LoRA effect")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image.")

MODEL_VARIANTS = {
    "default": "flux1-dev-F16.gguf",
    "q2_k": "flux1-dev-Q2_K.gguf",
    "q3_k_s": "flux1-dev-Q3_K_S.gguf",
    "q4_0": "flux1-dev-Q4_0.gguf",
    "q4_1": "flux1-dev-Q4_1.gguf",
    "q4_k_s": "flux1-dev-Q4_K_S.gguf",
    "q5_0": "flux1-dev-Q5_0.gguf",
    "q5_1": "flux1-dev-Q5_1.gguf",
    "q5_k_s": "flux1-dev-Q5_K_S.gguf",
    "q6_k": "flux1-dev-Q6_K.gguf",
    "q8_0": "flux1-dev-Q8_0.gguf",
}
DEFAULT_VARIANT = "default"

def get_civit_download_url(model_id):
    url = f"https://civitai.com/api/v1/models/{model_id}"
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
        """Initialize your model and resources here."""
        logging.basicConfig(level=logging.INFO)
        repo_id = "city96/FLUX.1-dev-gguf"
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        filename = MODEL_VARIANTS[variant]
        self.original_model_id = "black-forest-labs/FLUX.1-dev"
        logging.info(f"Downloading {filename} from {repo_id}...")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logging.info(f"Model downloaded to {ckpt_path}")
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        self.pipeline = FluxPipeline.from_pretrained(
            self.original_model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        lora_url = getattr(input_data, "lora_url", None)
        lora_multiplier = getattr(input_data, "lora_multiplier", 1.0)
        
        # Load LoRA adapter if provided
        if lora_url:
            success = load_lora_adapter(self.pipeline, lora_url, "user_lora", lora_multiplier)
            if success:
                # Set the adapter as active with the specified scale
                self.pipeline.set_adapters("user_lora")
        
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
        
        image = self.pipeline(
            prompt=prompt, 
            height=height,
            width=width,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/generated_image.png"
        image.save(output_path)
        return AppOutput(image_output=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        self.pipeline = None