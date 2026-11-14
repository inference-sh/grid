from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional
import torch
from diffusers import (
    ChromaPipeline
)
import os
import tempfile
import logging
import requests
from accelerate import Accelerator
from dateutil.parser import parse as parse_date
import re
import urllib.parse
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
    denoise: float = Field(default=1.0, ge=0.0, le=1.0, description="Denoising strength (0.0 to 1.0, where 1.0 is standard full denoising)")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt to steer away from undesired outputs.")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image.")

MODEL_VARIANTS = {}
DEFAULT_VARIANT = "default"

def get_civit_download_url(model_id):
    """The download URL of Civitai model model_id's latest Chroma version."""
    url = f"https://civitai.com/api/v1/models/{model_id}"
    response = requests.get(url, headers=civitai_auth(), timeout=30)
    response.raise_for_status()
    def get_date(item):
        date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
        try:
            return parse_date(date_str)
        except Exception:
            return parse_date('1970-01-01T00:00:00Z')
    versions = sorted(response.json()["modelVersions"], key=get_date, reverse=True)
    versions = [v for v in versions if v.get('baseModel') == 'Chroma']
    if not versions:
        raise ValueError(f"Civitai model {model_id} has no Chroma version")
    return versions[0]["downloadUrl"]

def civitai_auth():
    """The Authorization header for civitai.com when the user connected
    Civitai (credential civitai, CIVITAI_TOKEN), else none. The run holds a
    placeholder; the egress gateway puts the key in only toward civitai.com,
    so the header is sent there and nowhere else."""
    token = os.environ.get('CIVITAI_TOKEN')
    return {"Authorization": f"Bearer {token}"} if token else {}

def fetch_lora(url):
    """The LoRA file at url. A public file goes through the SDK's file cache
    (FILE_CACHE_DIR): fetched once per URL, reused by later tasks, expired
    by the engine. A Civitai file that needs the user's login is fetched
    with it into this task's scratch (TMPDIR), never into the shared cache:
    what one user's login unlocks is not served to another."""
    try:
        return File(uri=url).path
    except Exception:
        auth = civitai_auth()
        if not auth or (urllib.parse.urlsplit(url).hostname or "") != "civitai.com":
            raise
    # requests drops Authorization on the redirect to Civitai's CDN.
    with requests.get(url, headers=auth, stream=True, timeout=120) as response:
        response.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        with os.fdopen(fd, "wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
    return path

def lora_source(lora_url):
    """Where the LoRA at lora_url is: (a Hugging Face repo, the file in it,
    or None for the one diffusers finds) or (a local file, None)."""
    # Hugging Face file URL: https://huggingface.co/owner/repo/(blob|resolve)/<branch>/file.safetensors
    if "huggingface.co" in lora_url and ("/blob/" in lora_url or "/resolve/" in lora_url):
        parts = lora_url.split('/')
        marker = 'blob' if 'blob' in parts else 'resolve'
        if len(parts) < 7 or 'huggingface.co' not in parts or marker not in parts:
            raise ValueError(f"Invalid Hugging Face file URL: {lora_url}")
        at = parts.index(marker)
        return '/'.join(parts[parts.index('huggingface.co') + 1:at]), '/'.join(parts[at + 2:])
    # Hugging Face repo: "owner/repo" or "owner/repo/file.safetensors"
    if "/" in lora_url and not lora_url.startswith('http') and "civitai.com" not in lora_url:
        parts = lora_url.split('/')
        return '/'.join(parts[:2]), '/'.join(parts[2:]) or None
    if lora_url.endswith('.safetensors') and lora_url.startswith('http'):
        return fetch_lora(lora_url), None
    if "civitai.com" in lora_url:
        match = re.search(r"/models/(\d+)", lora_url)
        if not match:
            raise ValueError(f"No model ID in Civitai URL: {lora_url}")
        return fetch_lora(get_civit_download_url(match.group(1))), None
    raise ValueError(f"Unsupported LoRA URL: {lora_url} (use a Hugging Face repo or file, a .safetensors URL, or a Civitai model page)")

def comfy_to_kohya(state_dict):
    """ComfyUI and ai-toolkit save LoRAs under the original model's names
    (diffusion_model.double_blocks.0.img_attn.qkv) in PEFT form
    (lora_A/lora_B), which diffusers loads as no weights at all. diffusers
    maps the same names in kohya form (lora_down/lora_up), so this renames
    them to that. Anything else passes through."""
    if not any(k.startswith("diffusion_model.") and ".lora_A." in k for k in state_dict):
        return state_dict
    return {k.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up."): v for k, v in state_dict.items()}

def adapters(pipeline):
    return {name for names in pipeline.get_list_adapters().values() for name in names}

def load_lora_adapter(pipeline, lora_url, adapter_name, lora_multiplier=1.0):
    """Load the LoRA at lora_url into pipeline as adapter_name. Raises when
    it cannot be fetched or read, or none of its weights fit the model: the
    run asked for it, and an image made without it is not what was asked."""
    try:
        source, weight_name = lora_source(lora_url)
        state_dict, alphas = pipeline.lora_state_dict(source, weight_name=weight_name, return_alphas=True)
        pipeline.load_lora_weights(comfy_to_kohya({**state_dict, **(alphas or {})}), adapter_name=adapter_name)
        if adapter_name not in adapters(pipeline):
            raise ValueError("none of its weights match this model")
    except Exception as e:
        if adapter_name in adapters(pipeline):
            pipeline.delete_adapters(adapter_name)
        raise ValueError(f"LoRA '{adapter_name}' ({lora_url}) could not be loaded: {e or type(e).__name__}") from e
    logging.info(f"Loaded LoRA adapter '{adapter_name}' from {lora_url} (multiplier {lora_multiplier})")

class App(BaseApp):
    async def setup(self, metadata):
        # Initialize LoRA tracking attributes
        self.loaded_loras = {}  # adapter_name -> (lora_url, lora_multiplier)
        self.last_lora_url = None
        self.last_lora_multiplier = None
        self.last_lora_adapter_name = None
        
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading Chroma1-HD pipeline...")
        self.pipeline = ChromaPipeline.from_pretrained(
            "lodestones/Chroma1-HD",
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        loras = getattr(input_data, "loras", None) or []
        current_adapters = set(self.loaded_loras.keys())
        # requested_adapters set was unused; removed to avoid linter warning

        # Unload adapters that are no longer requested or changed
        for adapter_name in list(current_adapters):
            found = next((lora_cfg for lora_cfg in loras if lora_cfg.adapter_name == adapter_name), None)
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
                load_lora_adapter(self.pipeline, lora.lora_url, lora.adapter_name, lora.lora_multiplier)
                self.loaded_loras[lora.adapter_name] = (lora.lora_url, lora.lora_multiplier)
            active_adapters.append(lora.adapter_name)
            adapter_weights.append(lora.lora_multiplier)

        # Set all active adapters
        if active_adapters:
            self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)

        # For Chroma, keep default scheduler; log denoise only
        logging.info(f"Denoise: {input_data.denoise}")
        
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
            negative_prompt=getattr(input_data, "negative_prompt", None),
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