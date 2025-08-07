from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from diffusers import (
    FluxKontextPipeline, 
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)
from diffusers.utils import load_image
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
from transformers import (
    T5EncoderModel,
    QuantoConfig
)

from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_debug()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_url: str = Field(description="URL to LoRA file (.safetensors) or Civitai model page")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the LoRA effect")

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to edit the image with.")
    input_image: File = Field(description="The input image to be edited.")
    num_inference_steps: int = Field(default=30, description="The number of inference steps.")
    guidance_scale: float = Field(default=2.5, description="The guidance scale.")
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image.")

MODEL_VARIANTS = {
    "default": "flux1-kontext-dev-BF16.gguf",
    "q2_k": "flux1-kontext-dev-Q2_K.gguf",
    "q3_k_s": "flux1-kontext-dev-Q3_K_S.gguf",
    "q4_k_m": "flux1-kontext-dev-Q4_K_M.gguf",
    "q4_k_s": "flux1-kontext-dev-Q4_K_S.gguf",
    "q5_k_m": "flux1-kontext-dev-Q5_K_M.gguf",
    "q5_k_s": "flux1-kontext-dev-Q5_K_S.gguf",
    "q6_k": "flux1-kontext-dev-Q6_K.gguf",
    "q8_0": "flux1-kontext-dev-Q8_0.gguf",
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
        repo_id = "bullerwins/FLUX.1-Kontext-dev-GGUF"
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        filename = MODEL_VARIANTS[variant]
        logging.info(f"Downloading {filename} from {repo_id}...")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logging.info(f"Model downloaded to {ckpt_path}")
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config="black-forest-labs/FLUX.1-Kontext-dev",
            subfolder="transformer",
        )
        quanto_config = None
        if variant == "q2_k":
            quanto_config = QuantoConfig(weights="int2")
        elif variant == "q3_k_s":
            quanto_config = QuantoConfig(weights="int8")
       

        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            quantization_config=quanto_config,
        )

        self.pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16
        )
        # self.pipeline.to(device)
        # self.pipeline.transformer.set_attention_backend("sage_varlen")
        self.pipeline.transformer.to(memory_format=torch.channels_last)
        # self.pipeline.transformer.compile_repeated_blocks(fullgraph=True)
        # self.pipeline.transformer = torch.compile(
        #     self.pipeline.transformer, mode="max-autotune", fullgraph=True
        # )
        # self.pipeline.vae.decode = torch.compile(
        #     self.pipeline.vae.decode, mode="max-autotune", fullgraph=True
        # )
        self.pipeline.enable_model_cpu_offload()
        # self.pipeline.enable_sequential_cpu_offload()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        loras = getattr(input_data, "loras", None) or []
        current_adapters = set(self.loaded_loras.keys())
        requested_adapters = set(lora.adapter_name for lora in loras)

        # Unload adapters that are no longer requested or changed
        for adapter_name in list(current_adapters):
            found = next((l for l in loras if l.adapter_name == adapter_name), None)
            if not found or (
                self.loaded_loras[adapter_name][0] != found.lora_url or
                self.loaded_loras[adapter_name][1] != found.lora_multiplier
            ):
                try:
                    self.pipeline.delete_adapters(adapter_name)
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
                self.loaded_loras[lora.adapter_name][1] != lora.lora_multiplier
            ):
                success = load_lora_adapter(self.pipeline, lora.lora_url, lora.adapter_name, lora.lora_multiplier)
                if success:
                    self.loaded_loras[lora.adapter_name] = (lora.lora_url, lora.lora_multiplier)
            if lora.adapter_name in self.loaded_loras:
                active_adapters.append(lora.adapter_name)
                adapter_weights.append(lora.lora_multiplier)

        # Set all active adapters
        if active_adapters:
            self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)

        prompt = input_data.prompt
        input_image_path = input_data.input_image.path
        num_inference_steps = input_data.num_inference_steps
        guidance_scale = input_data.guidance_scale
        seed = input_data.seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        image = load_image(input_image_path).convert("RGB")
        width, height = image.size
        logging.info(f"Editing image with prompt: '{prompt}' and size: {width}x{height}")
        result = self.pipeline(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            width=width,
            height=height,
        ).images[0]
        print("pipeline done")
        output_path = "/tmp/generated_image.png"
        result.save(output_path)
        return AppOutput(image_output=File(path=output_path))

    async def unload(self):
        self.pipeline = None