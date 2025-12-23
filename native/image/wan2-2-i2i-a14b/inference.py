import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import Field, BaseModel
from PIL import Image
import logging
import requests
import re
from datetime import datetime

from diffusers import WanTransformer3DModel, GGUFQuantizationConfig
from diffusers.hooks import FirstBlockCacheConfig, apply_group_offloading
from huggingface_hub import hf_hub_download
from accelerate import Accelerator

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Import our clean pipeline implementation
from .wan_upscaler import WanLowNoiseUpscalePipeline

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_file: File = Field(description="LoRA weights file (.safetensors)")
    lora_multiplier: float = Field(default=1.0, ge=0.0, le=10.0, description="Multiplier for the LoRA effect")

def get_civit_download_url(model_id):
    civitai_token = os.environ.get('CIVITAI_TOKEN')
    if civitai_token:
        url = f"https://civitai.com/models/{model_id}&token={civitai_token}"
    else:
        url = f"https://civitai.com/models/{model_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    versions = data["modelVersions"]

    def get_date(item):
        date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return datetime.fromisoformat(date_str)

    sorted_data = sorted(versions, key=get_date, reverse=True)
    wan_versions = [v for v in sorted_data if v.get('baseModel', '').lower().startswith('wan')]
    latest_version = wan_versions[0] if wan_versions else sorted_data[0]
    downloadUrl = latest_version["downloadUrl"]
    return downloadUrl

def download_model_data(model_id, downloadUrl, targetFolder):
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    filePath = os.path.join(targetFolder, f"{model_id}.safetensors")
    if not os.path.isfile(filePath):
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

def load_lora_adapter(transformer, lora_source, adapter_name="lora", lora_multiplier=1.0):
    """Load a LoRA adapter for the low-noise transformer."""
    if not lora_source:
        return False
    
    def _load_single_lora(load_kwargs, base_adapter_name: str):
        try:
            if hasattr(transformer, 'load_lora_weights'):
                load_kwargs["adapter_name"] = base_adapter_name
                transformer.load_lora_weights(**load_kwargs)
                logging.info(f"Loaded LoRA adapter {base_adapter_name} onto low-noise transformer")
                return base_adapter_name
        except Exception as e:
            logging.error(f"Failed to load LoRA {base_adapter_name}: {e}")
        return None
    
    # Local file path
    if isinstance(lora_source, str) and os.path.isfile(lora_source):
        load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_source}
        created = _load_single_lora(load_kwargs, adapter_name)
        if created:
            logging.info(f"Loaded LoRA adapter {created} from local file: {lora_source}")
            return [created]
        return []

    lora_url = lora_source

    # Hugging Face blob URL
    if isinstance(lora_url, str) and "huggingface.co" in lora_url and "/blob/" in lora_url:
        parts = lora_url.split('/')
        if len(parts) >= 7 and 'huggingface.co' in parts and 'blob' in parts:
            repo_start = parts.index('huggingface.co') + 1
            blob_index = parts.index('blob')
            repo_id = '/'.join(parts[repo_start:blob_index])
            weight_name = '/'.join(parts[blob_index + 2:])
            load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
            created = _load_single_lora(load_kwargs, adapter_name)
            if created:
                logging.info(f"Loaded LoRA adapter {created} from HF repo: {repo_id}, file: {weight_name}")
                return [created]
            return []
        else:
            raise ValueError(f"Invalid Hugging Face blob URL format: {lora_url}")
    
    # Hugging Face resolve URL
    elif isinstance(lora_url, str) and "huggingface.co" in lora_url and "/resolve/" in lora_url:
        parts = lora_url.split('/')
        if len(parts) >= 7 and 'huggingface.co' in parts and 'resolve' in parts:
            repo_start = parts.index('huggingface.co') + 1
            resolve_index = parts.index('resolve')
            repo_id = '/'.join(parts[repo_start:resolve_index])
            weight_name = '/'.join(parts[resolve_index + 2:])
            load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
            created = _load_single_lora(load_kwargs, adapter_name)
            if created:
                logging.info(f"Loaded LoRA adapter {created} from HF repo: {repo_id}, file: {weight_name}")
                return [created]
            return []
        else:
            raise ValueError(f"Invalid Hugging Face resolve URL format: {lora_url}")
    
    # Hugging Face repository string
    elif isinstance(lora_url, str) and "/" in lora_url and not lora_url.startswith('http') and "civitai.com" not in lora_url:
        parts = lora_url.split('/')
        if len(parts) == 2:
            repo_id = lora_url
            load_kwargs = {"repo_id": repo_id}
            created = _load_single_lora(load_kwargs, adapter_name)
            if created:
                logging.info(f"Loaded LoRA adapter {created} from HF repo: {repo_id}")
                return [created]
            return []
        elif len(parts) > 2:
            repo_id = '/'.join(parts[:2])
            weight_name = '/'.join(parts[2:])
            load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
            created = _load_single_lora(load_kwargs, adapter_name)
            if created:
                logging.info(f"Loaded LoRA adapter {created} from HF repo: {repo_id}, file: {weight_name}")
                return [created]
            return []
    
    # Direct .safetensors URL
    elif isinstance(lora_url, str) and lora_url.endswith('.safetensors') and lora_url.startswith('http'):
        lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)
        
        model_id = os.path.splitext(os.path.basename(lora_url))[0]
        lora_path = download_model_data(model_id, lora_url, lora_dir)
        load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_path}
        created = _load_single_lora(load_kwargs, adapter_name)
        if created:
            logging.info(f"Loaded LoRA adapter {created} from URL: {lora_url}")
            return [created]
        return []
    
    # Civitai model URL
    elif isinstance(lora_url, str) and "civitai.com" in lora_url:
        match = re.search(r"/models/(\d+)", lora_url)
        if not match:
            raise ValueError("Could not extract model ID from Civitai URL")
        
        model_id = match.group(1)
        download_url = get_civit_download_url(model_id)
        if not download_url:
            raise RuntimeError(f"No download URL found for Civitai model {model_id}")
        
        lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)
        
        lora_path = download_model_data(model_id, download_url, lora_dir)
        load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_path}
        created = _load_single_lora(load_kwargs, adapter_name)
        if created:
            logging.info(f"Loaded LoRA adapter {created} from Civitai model: {model_id}")
            return [created]
        return []
    
    else:
        raise ValueError(f"Unsupported LoRA source format: {lora_url}")

# Model variants mapping for GGUF quantization from QuantStack
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "q2_k": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf",
    "q3_k_s": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf",
    "q3_k_m": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_M.gguf",
    "q4_0": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_0.gguf",
    "q4_1": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_1.gguf",
    "q4_k_s": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf",
    "q4_k_m": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
    "q5_0": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_0.gguf",
    "q5_1": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_1.gguf",
    "q5_k_s": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_S.gguf",
    "q5_k_m": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_M.gguf",
    "q6_k": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q6_K.gguf",
    "q8_0": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"
}

DEFAULT_VARIANT = "default"

class AppInput(BaseAppInput):
    image: File = Field(description="Input image to upscale")
    prompt: str = Field(default="", description="Text prompt to guide upscaling")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid"
    )
    scale: Optional[float] = Field(default=1.0, description="Upscale factor (if not specifying width/height)")
    width: Optional[int] = Field(default=None, description="Target width (overrides scale)")
    height: Optional[int] = Field(default=None, description="Target height (overrides scale)")
    guidance_scale: float = Field(default=2.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=40, description="Number of denoising steps")
    strength: float = Field(default=0.3, description="Amount of noise to add (0.0-1.0)")
    sharpen_input: float = Field(default=0.0, description="Sharpen input image (0.0-1.0+)")
    desaturate_input: float = Field(default=0.0, description="Desaturate input image (0.0-1.0)")
    pre_downscale_factor: float = Field(default=1.0, ge=0.1, le=1.0, description="Pre-downscale factor (0.1-1.0). Values < 1.0 downscale the image first, then upscale to target size with latent noise generation. Lower values add more details/creativity by creating more missing information for the transformer to fill in.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0, description="Cache threshold for transformer (0 to disable caching)")
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")
    
class AppOutput(BaseAppOutput):
    image: File = Field(description="Upscaled output image")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the WAN 2.2 Low-Noise Upscaler pipeline."""
        print("Setting up WAN 2.2 Low-Noise Upscaler pipeline...")
        
        # Initialize LoRA tracking attributes
        self.loaded_loras = {}  # adapter_name -> (lora_source_id, lora_multiplier, created_names)
        
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
        
        # Determine offloading strategy from variant suffix
        use_cpu_offload = variant.endswith("_offload")
        use_group_offload = variant.endswith("_offload_lowvram")

        # Strip offloading suffix for base variant lookup
        base_variant = variant.replace("_offload_lowvram", "").replace("_offload", "")
        
        if base_variant not in MODEL_VARIANTS:
            print(f"Unknown variant '{base_variant}', falling back to default '{DEFAULT_VARIANT}'")
            base_variant = DEFAULT_VARIANT
        
        print(f"Loading model variant: {base_variant} (offload: cpu={use_cpu_offload}, group={use_group_offload})")

        # Build transformer externally (GGUF quantized or regular) and pass into pipeline
        transformer = None
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        quantized_repo = "QuantStack/Wan2.2-T2V-A14B-GGUF"

        if base_variant != "default":
            gguf_file = MODEL_VARIANTS.get(base_variant)
            if gguf_file is None:
                print(f"Unknown variant '{base_variant}', falling back to default transformer")
                transformer = WanTransformer3DModel.from_pretrained(
                    model_id, subfolder="transformer_2", torch_dtype=self.dtype
                )
            else:
                low_path = hf_hub_download(repo_id=quantized_repo, filename=gguf_file)
                qcfg = GGUFQuantizationConfig(compute_dtype=self.dtype)
                transformer = WanTransformer3DModel.from_single_file(
                    low_path,
                    quantization_config=qcfg,
                    config=model_id,
                    subfolder="transformer_2",
                    torch_dtype=self.dtype,
                )
        else:
            transformer = WanTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer_2", torch_dtype=self.dtype
            )

        self.pipe = WanLowNoiseUpscalePipeline.from_pretrained(
            device=self.device,
            dtype=self.dtype,
            transformer=transformer,
        )
                
        # Apply offloading after pipeline creation
        if use_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload")
        elif use_group_offload:
            onload_device = self.device
            offload_device = torch.device("cpu")
            
            # Apply group offloading to each component
            self.pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            self.pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            print(f"Enabled group offloading (GPU: {onload_device} <-> CPU: {offload_device})")
        else:
            # No offloading: move to accelerator device
            self.pipe = self.pipe.to(self.device)
            print("No offloading enabled, pipeline on GPU")
        
        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Upscale image using WAN 2.2 low-noise transformer."""
        print(f"Upscaling image with prompt: {input_data.prompt}")
        
        # Load input image
        image = Image.open(input_data.image.path).convert("RGB")
        print(f"Input image size: {image.size}")
        
        # Configure caching if threshold is non-zero
        if hasattr(self.pipe.transformer, 'disable_cache'):
            self.pipe.transformer.disable_cache()
        
        if input_data.cache_threshold > 0:
            print(f"Enabling cache for transformer with threshold: {input_data.cache_threshold}")
            cache_config = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipe.transformer.enable_cache(cache_config)
        
        # Handle LoRA adapters
        loras = getattr(input_data, "loras", None) or []
        requested_by_name = {l.adapter_name: l for l in loras}

        # Unload adapters that are no longer requested or whose config changed
        for adapter_name in list(self.loaded_loras.keys()):
            previous_source, previous_mult, previous_created = self.loaded_loras[adapter_name]
            found = requested_by_name.get(adapter_name)
            if (
                found is None
                or previous_source != found.lora_file.path
                or previous_mult != found.lora_multiplier
            ):
                for created_name in previous_created:
                    if hasattr(self.pipe.transformer, 'delete_adapters'):
                        self.pipe.transformer.delete_adapters(created_name)
                    logging.info(f"Unloaded previous LoRA adapter: {created_name}")
                del self.loaded_loras[adapter_name]

        # Load requested adapters
        for lora in loras:
            needs_load = (
                lora.adapter_name not in self.loaded_loras
                or self.loaded_loras[lora.adapter_name][0] != lora.lora_file.path
                or self.loaded_loras[lora.adapter_name][1] != lora.lora_multiplier
            )
            if needs_load:
                created_names = load_lora_adapter(
                    self.pipe.transformer,
                    lora.lora_file.path,
                    lora.adapter_name,
                    lora.lora_multiplier,
                )
                if created_names:
                    self.loaded_loras[lora.adapter_name] = (
                        lora.lora_file.path,
                        lora.lora_multiplier,
                        created_names,
                    )

        # Activate all requested adapters
        active_adapters = []
        adapter_weights = []
        for lora in loras:
            if lora.adapter_name in self.loaded_loras:
                created_names = self.loaded_loras[lora.adapter_name][2]
                for created_name in created_names:
                    active_adapters.append(created_name)
                    adapter_weights.append(lora.lora_multiplier)

        if active_adapters and hasattr(self.pipe.transformer, 'set_adapters'):
            self.pipe.transformer.set_adapters(active_adapters, adapter_weights=adapter_weights)
            logging.info(f"Activated LoRA adapters: {active_adapters} with weights: {adapter_weights}")
        
        # Set seed if provided
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(input_data.seed)
        
        # Upscale image
        print("Starting upscaling...")
        result = self.pipe(
            image=image,
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            scale=input_data.scale,
            width=input_data.width,
            height=input_data.height,
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
            strength=input_data.strength,
            sharpen_input=input_data.sharpen_input,
            desaturate_input=input_data.desaturate_input,
            pre_downscale_factor=input_data.pre_downscale_factor,
        )
        
        print("Upscaling complete, exporting...")
        
        # Create temporary file for image output
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Save result
        result.save(output_path)
        print(f"Upscaled image size: {result.size}")
        print(f"Image exported to: {output_path}")
        
        return AppOutput(image=File(path=output_path))

