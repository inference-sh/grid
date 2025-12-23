from typing import Optional
from pydantic import Field, BaseModel
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import StableDiffusionXLPipeline
import torch
import logging
import os
import requests
import re
from datetime import datetime

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_url: str = Field(description="URL to LoRA file (.safetensors) or Civitai model page")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the LoRA effect")

# CPU Offloading variants for different memory configurations
OFFLOADING_VARIANTS = {
    "full_gpu": "Full GPU",
    "model_offload": "Model CPU Offload", 
    "sequential_offload": "Sequential CPU Offload"
}

DEFAULT_VARIANT = "model_offload"

def get_civit_download_url(model_id):
    # Use the proper Civitai API endpoint
    url = f"https://civitai.com/api/v1/models/{model_id}"
    
    # Set headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
    }
    
    # Add Civitai token if available
    civitai_token = os.environ.get('CIVITAI_TOKEN')
    if civitai_token:
        headers['Authorization'] = f'Bearer {civitai_token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'modelVersions' not in data or not data['modelVersions']:
            logging.error(f"No model versions found for Civitai model {model_id}")
            return None
            
        versions = data["modelVersions"]
        
        def get_date(item):
            date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
            try:
                # Try parsing ISO format first
                if 'T' in date_str and 'Z' in date_str:
                    # Remove 'Z' and parse as ISO format
                    date_str = date_str.replace('Z', '+00:00')
                    return datetime.fromisoformat(date_str)
                elif 'T' in date_str:
                    return datetime.fromisoformat(date_str)
                else:
                    # Fallback to strptime for other formats
                    return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
            except Exception as e:
                return datetime(1970, 1, 1)
                
        sorted_data = sorted(versions, key=get_date, reverse=True)
        
        # Filter for SDXL compatible models (be more flexible with baseModel matching)
        compatible_versions = []
        for v in sorted_data:
            base_model = v.get('baseModel', '').lower()
            if any(keyword in base_model for keyword in ['sdxl', 'sd xl', 'stable diffusion xl']):
                compatible_versions.append(v)
        
        # If no SDXL-specific versions found, try the latest version anyway
        if not compatible_versions:
            logging.warning(f"No SDXL-specific versions found for model {model_id}, using latest version")
            compatible_versions = sorted_data
            
        if not compatible_versions:
            logging.error(f"No compatible versions found for Civitai model {model_id}")
            return None

        latest_version = compatible_versions[0]
        
        # Look for downloadUrl in the version or its files
        download_url = latest_version.get("downloadUrl")
        if not download_url and 'files' in latest_version:
            # Try to find a safetensors file
            for file_info in latest_version['files']:
                if file_info.get('name', '').endswith('.safetensors'):
                    download_url = file_info.get('downloadUrl')
                    break
        
        if not download_url:
            logging.error(f"No download URL found in Civitai model {model_id} version data")
            return None
            
        return download_url
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch Civitai model info (Network error): {e}")
        return None
    except KeyError as e:
        logging.error(f"Unexpected Civitai API response format for model {model_id}: missing {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to fetch Civitai model info for model {model_id}: {e}")
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

class AppInput(BaseAppInput):
    prompt: str = Field(
        ...,
        description="The text prompt to generate the image from",
        examples=["A majestic lion jumping from a big stone at night"]
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Negative prompt to avoid certain elements in the generated image",
        examples=["blurry, low quality, distorted"]
    )
    num_inference_steps: int = Field(
        50,
        description="Number of denoising steps",
        ge=1,
        le=100
    )
    guidance_scale: float = Field(
        7.5,
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
    model_url: str = Field(
        "stabilityai/stable-diffusion-xl-base-1.0",
        description="URL or huggingface path to a custom Stable Diffusion XL model",
        examples=["stabilityai/stable-diffusion-xl-base-1.0"]
    )
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    pipeline: Optional[StableDiffusionXLPipeline] = None
    default_model_url: str = "stabilityai/stable-diffusion-xl-base-1.0"
    default_variant: Optional[str] = None

    async def setup(self, metadata):
        """Initialize the Stable Diffusion XL model with appropriate offloading strategy."""
        # Initialize LoRA tracking attributes
        self.loaded_loras = {}  # adapter_name -> (lora_url, lora_multiplier)
        
        logging.basicConfig(level=logging.INFO)
        
        # Get offloading strategy from metadata
        offloading_strategy = getattr(metadata, "app_variant", self.default_variant)
        
        if offloading_strategy not in OFFLOADING_VARIANTS:
            logging.warning(f"Unknown offloading strategy '{offloading_strategy}', falling back to default '{self.default_variant}'")
            offloading_strategy = self.default_variant
        
        logging.info(f"Loading SDXL with {OFFLOADING_VARIANTS[offloading_strategy]}")
        
        # Load the base pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.default_model_url,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Apply the appropriate offloading strategy
        self._apply_offloading_strategy(offloading_strategy)
        
        logging.info(f"Successfully loaded SDXL with {offloading_strategy} strategy")

    def _apply_offloading_strategy(self, strategy: str):
        """Apply the specified CPU offloading strategy."""
        if strategy == "full_gpu":
            # Full GPU - everything stays on GPU
            logging.info("Applying full GPU strategy - moving all components to CUDA")
            self.pipeline.to("cuda")
            
        elif strategy == "model_offload":
            # Model CPU offload - automatic GPU/CPU memory management
            logging.info("Applying model CPU offload - enabling automatic memory management")
            self.pipeline.enable_model_cpu_offload()
            
        elif strategy == "sequential_offload":
            # Sequential CPU offload - moves models to GPU sequentially as needed
            logging.info("Applying sequential CPU offload - components loaded sequentially")
            self.pipeline.enable_sequential_cpu_offload()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image based on the input prompt."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Handle custom model URL
        if input_data.model_url != self.default_model_url:
            # Load custom model URL (regular model)
            logging.info(f"Loading custom model: {input_data.model_url}")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                input_data.model_url,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.pipeline.to("cuda")

        # Handle LoRA adapters
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

        # Generate the image
        image = self.pipeline(
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            width=input_data.width,
            height=input_data.height
        ).images[0]

        # Save the image
        output_path = "/tmp/generated_image.png"
        image.save(output_path)

        return AppOutput(result=File(path=output_path))

