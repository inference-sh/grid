import os
import sys

# Add ControlNetPlus folder to Python path
controlnet_plus_path = os.path.join(os.path.dirname(__file__), "ControlNetPlus")
sys.path.append(controlnet_plus_path)

from typing import Optional, List
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxTransformer2DModel, GGUFQuantizationConfig
from controlnet_aux import ZoeDetector
import torch
import cv2
import numpy as np
from PIL import Image
from enum import Enum
from huggingface_hub import hf_hub_download
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

# Model variants for GGUF quantization
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

class ControlNetType(str, Enum):
    CANNY = "canny"  # 0
    TILE = "tile"    # 1
    DEPTH = "depth"  # 2
    BLUR = "blur"    # 3
    POSE = "pose"    # 4
    GRAY = "gray"    # 5
    LOW_QUALITY = "low_quality"  # 6

    def to_int(self) -> int:
        """Convert ControlNetType to its corresponding integer value."""
        mapping = {
            ControlNetType.CANNY: 0,
            ControlNetType.TILE: 1,
            ControlNetType.DEPTH: 2,
            ControlNetType.BLUR: 3,
            ControlNetType.POSE: 4,
            ControlNetType.GRAY: 5,
            ControlNetType.LOW_QUALITY: 6
        }
        return mapping[self]

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

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

class AppInput(BaseAppInput):
    prompt: str = Field(
        ...,
        description="The text prompt to generate the image from",
        examples=["A majestic lion jumping from a big stone at night"]
    )
    num_inference_steps: int = Field(
        28,
        description="Number of denoising steps",
        ge=1,
        le=100
    )
    guidance_scale: float = Field(
        3.5,
        description="Classifier-free guidance scale",
        ge=1.0,
        le=20.0
    )
    width: int = Field(
        1024,
        description="Width of the generated image",
        ge=256,
        le=2048
    )
    height: int = Field(
        1024,
        description="Height of the generated image",
        ge=256,
        le=2048
    )
    seed: Optional[int] = Field(
        default=None, 
        description="The seed for random generation."
    )
    controlnet_type: ControlNetType = Field(
        None,
        description="Type of ControlNet to use"
    )
    controlnet_image: File = Field(
        None,
        description="Input image for the ControlNet"
    )
    controlnet_strength: float = Field(
        1.0,
        description="Strength of the ControlNet effect",
        ge=0.0,
        le=1.0
    )
    controlnet_pre_process: bool = Field(
        True,
        description="Whether to pre-process the input image",
    )
    control_guidance_start: float = Field(
        0.0,
        description="When to start applying ControlNet guidance (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    control_guidance_end: float = Field(
        1.0,
        description="When to stop applying ControlNet guidance (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    lora_url: Optional[str] = Field(
        default=None, 
        description="URL to LoRA file (.safetensors) or Civitai model page"
    )
    lora_multiplier: float = Field(
        default=1.0, 
        description="Multiplier for the LoRA effect"
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    pipeline: Optional[FluxControlNetPipeline] = None
    processor: Optional[ZoeDetector] = None

    async def setup(self, metadata):
        """Initialize the FLUX model with ControlNet support and GGUF quantization."""
        logging.basicConfig(level=logging.INFO)
        
        # Initialize depth processor
        self.processor = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        # Setup GGUF quantized transformer
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

        # Initialize ControlNet model
        controlnet_model = FluxControlNetModel.from_pretrained(
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )

        # Create pipeline with quantized transformer and ControlNet
        self.pipeline = FluxControlNetPipeline.from_pretrained(
            self.original_model_id,
            transformer=transformer,
            controlnet=controlnet_model,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image based on the input prompt and ControlNet."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Load LoRA adapter if provided
        lora_url = getattr(input_data, "lora_url", None)
        lora_multiplier = getattr(input_data, "lora_multiplier", 1.0)
        
        if lora_url:
            success = load_lora_adapter(self.pipeline, lora_url, "user_lora", lora_multiplier)
            if success:
                # Set the adapter as active with the specified scale
                self.pipeline.set_adapters("user_lora")

        # Process ControlNet input if provided
        control_image = None
        control_mode = None
        control_scale = None

        if input_data.controlnet_type and input_data.controlnet_image:
            # Read the image
            img = cv2.imread(input_data.controlnet_image.path)
            
            # If pre_process is False, just resize the image and return
            if not input_data.controlnet_pre_process:
                height, width, _ = img.shape
                ratio = np.sqrt(1024. * 1024. / (width * height))
                new_width, new_height = int(width * ratio), int(height * ratio)
                processed_img = cv2.resize(img, (new_width, new_height))
                control_image = Image.fromarray(processed_img)
            else:
                # Process based on ControlNet type
                if input_data.controlnet_type == ControlNetType.CANNY:
                    processed_img = cv2.Canny(img, 100, 200)
                    processed_img = HWC3(processed_img)
                elif input_data.controlnet_type == ControlNetType.DEPTH:
                    processed_img = self.processor(img, output_type='cv2')
                elif input_data.controlnet_type == ControlNetType.POSE:
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    processed_img = HWC3(processed_img)
                elif input_data.controlnet_type == ControlNetType.GRAY:
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    processed_img = HWC3(processed_img)
                elif input_data.controlnet_type == ControlNetType.BLUR:
                    processed_img = cv2.GaussianBlur(img, (5, 5), 0)
                elif input_data.controlnet_type == ControlNetType.TILE:
                    processed_img = img
                elif input_data.controlnet_type == ControlNetType.LOW_QUALITY:
                    height, width = img.shape[:2]
                    processed_img = cv2.resize(img, (width//4, height//4))
                    processed_img = cv2.resize(processed_img, (width, height))
                else:
                    processed_img = img

                # Resize the image
                height, width, _ = processed_img.shape
                ratio = np.sqrt(1024. * 1024. / (width * height))
                new_width, new_height = int(width * ratio), int(height * ratio)
                processed_img = cv2.resize(processed_img, (new_width, new_height))
                
                control_image = Image.fromarray(processed_img)

            control_mode = input_data.controlnet_type.to_int()
            control_scale = input_data.controlnet_strength

        # Generate the image
        seed = input_data.seed
        generator = None
        
        # Use accelerator.device for generator device
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = torch.Generator(device).manual_seed(torch.randint(0, 2147483647, (1,)).item())
        
        print("Control Image: ", control_image)
        print("Mode: ", control_mode)
        print("Scale: ", control_scale)

        images = self.pipeline(
            prompt=input_data.prompt,
            control_image=control_image,
            control_mode=control_mode,
            generator=generator,
            width=input_data.width,
            height=input_data.height,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            control_guidance_start=input_data.control_guidance_start,
            control_guidance_end=input_data.control_guidance_end,
            controlnet_conditioning_scale=control_scale,
        ).images

        # Save the image
        output_path = "/tmp/generated_image.png"
        images[0].save(output_path)

        return AppOutput(result=File(path=output_path))
