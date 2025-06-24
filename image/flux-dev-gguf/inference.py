from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, ConfigDict
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from.", default="A cat astronaut, photorealistic, 4k")
    height: int = Field(1024, description="The height in pixels of the generated image.")
    width: int = Field(1024, description="The width in pixels of the generated image.")
    num_inference_steps: int = Field(description="The number of inference steps.", default=30)
    guidance_scale: float = Field(description="The guidance scale.", default=3.5)
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")
    lora_url: Optional[str] = Field(default=None, description="URL to LoRA file (.safetensors) or Civitai model page")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the LoRA effect")

class AppOutput(BaseAppOutput):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }
    )
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
    response = requests.get(url)
    data = response.json()
    versions = data["modelVersions"]
    def get_date(item):
        return item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
    sorted_data = sorted(versions, key=lambda x: datetime.strptime(get_date(x), '%Y-%m-%dT%H:%M:%S.%f'), reverse=True)
    latest_version = sorted_data[0]
    downloadUrl = latest_version["downloadUrl"]
    return downloadUrl

def download_model_data(model_id, downloadUrl, targetFolder):
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    filePath = os.path.join(targetFolder, f"{model_id}.safetensors")
    if not os.path.isfile(filePath):
        response = requests.get(downloadUrl)
        with open(filePath, 'wb') as file:
            file.write(response.content)
    return filePath

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
        # LoRA support (now handled at run time)
        lora_url = getattr(input_data, "lora_url", None)
        lora_multiplier = getattr(input_data, "lora_multiplier", 1.0)
        lora_path = None
        if lora_url:
            # Do not create directories, just use 'loras' if it exists, else fallback to '/tmp/loras' or '/tmp'
            lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras" if os.path.isdir("/tmp/loras") else "/tmp"
            # 1. Direct safetensors URL
            if lora_url.endswith('.safetensors') and (lora_url.startswith('http://') or lora_url.startswith('https://')):
                model_id = os.path.splitext(os.path.basename(lora_url))[0]
                lora_path = download_model_data(model_id, lora_url, lora_dir)
            # 2. Civitai URL
            elif "civitai.com" in lora_url:
                import re
                match = re.search(r"/models/(\d+)", lora_url)
                if match:
                    model_id = match.group(1)
                    download_url = get_civit_download_url(model_id)
                    lora_path = download_model_data(model_id, download_url, lora_dir)
                else:
                    logging.warning("Could not extract model id from civitai url")
                    lora_path = None
            # 3. Hugging Face repo string (e.g., 'fofr/flux-80s-cyberpunk' or 'fofr/flux-80s-cyberpunk/filename.safetensors')
            elif "/" in lora_url and not lora_url.startswith('http'):
                # If a filename is provided, use it; otherwise, try to infer
                parts = lora_url.split('/')
                if len(parts) == 2:
                    repo_id = lora_url
                    # Try to infer filename: use repo name + '.safetensors'
                    filename = parts[1] + '.safetensors'
                elif len(parts) > 2:
                    repo_id = '/'.join(parts[:2])
                    filename = '/'.join(parts[2:])
                else:
                    logging.warning("Invalid Hugging Face repo string for LoRA: %s", lora_url)
                    repo_id = None
                    filename = None
                if repo_id and filename:
                    try:
                        lora_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=lora_dir)
                    except Exception as e:
                        logging.warning(f"Failed to download LoRA from Hugging Face: {e}")
                        lora_path = None
            else:
                logging.warning("Unsupported lora_url format: %s", lora_url)
                lora_path = None
            if lora_path:
                try:
                    if hasattr(self.pipeline, "load_lora_weights"):
                        self.pipeline.load_lora_weights(lora_path, lora_scale=lora_multiplier)
                        logging.info(f"Loaded LoRA weights from {lora_path}")
                    else:
                        logging.warning("Pipeline does not support loading LoRA weights directly.")
                except Exception as e:
                    logging.warning(f"Failed to load LoRA: {e}")
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