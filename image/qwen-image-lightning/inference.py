from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional, Literal
import requests
import re
from datetime import datetime
import torch
import math
from huggingface_hub import hf_hub_download
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel, 
    GGUFQuantizationConfig,
    FirstBlockCacheConfig,
    UniPCMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.hooks import apply_group_offloading
import os
import logging

# Set up HuggingFace transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    qwen_versions = [v for v in sorted_data if v.get('baseModel', '').lower().startswith('qwen')]
    latest_version = qwen_versions[0] if qwen_versions else sorted_data[0]
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

def load_lora_adapter(pipeline, lora_source, adapter_name="lora", lora_multiplier=1.0):
    """Load a LoRA adapter for the Qwen-Image pipeline or transformer."""
    if not lora_source:
        return False
    
    def _load_single_lora(load_kwargs, base_adapter_name: str):
        # Try loading on transformer first
        try:
            transformer = getattr(pipeline, 'transformer', None)
            if transformer and hasattr(transformer, 'load_lora_weights'):
                print(f"🔄 Loading LoRA adapter '{base_adapter_name}' onto transformer...")
                load_kwargs["adapter_name"] = base_adapter_name
                transformer.load_lora_weights(**load_kwargs)
                print(f"✅ LoRA adapter '{base_adapter_name}' loaded successfully onto transformer")
                logging.info(f"Loaded LoRA adapter {base_adapter_name} onto Qwen-Image transformer")
                return base_adapter_name
        except Exception as e:
            print(f"⚠️ Failed to load LoRA on transformer: {e}, trying pipeline...")
            logging.warning(f"Failed to load LoRA {base_adapter_name} on transformer: {e}")
        
        # Fallback to pipeline
        try:
            if hasattr(pipeline, 'load_lora_weights'):
                print(f"🔄 Loading LoRA adapter '{base_adapter_name}' onto pipeline...")
                load_kwargs["adapter_name"] = base_adapter_name
                pipeline.load_lora_weights(**load_kwargs)
                print(f"✅ LoRA adapter '{base_adapter_name}' loaded successfully onto pipeline")
                logging.info(f"Loaded LoRA adapter {base_adapter_name} onto Qwen-Image pipeline")
                return base_adapter_name
            else:
                print(f"❌ Neither transformer nor pipeline support load_lora_weights method")
        except Exception as e:
            print(f"❌ Failed to load LoRA '{base_adapter_name}' on pipeline: {e}")
            logging.error(f"Failed to load LoRA {base_adapter_name} on pipeline: {e}")
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

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from. Supports both English and Chinese text rendering.")
    negative_prompt: Optional[str] = Field(default="", description="The negative prompt to guide what not to include in the image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    num_inference_steps: int = Field(default=8, description="The number of inference steps for generation quality.")
    true_cfg_scale: float = Field(default=1.0, description="The CFG scale for generation guidance.")
    seed: Optional[int] = Field(default=None, description="The seed for reproducible generation.")
    language: Optional[Literal["en", "zh"]] = Field(default="en", description="Language for prompt optimization (English or Chinese).")
    cache_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="First-block cache threshold for transformer (0 disables caching).")
    use_unipcm_flow_matching: bool = Field(default=False, description="If true, switch scheduler to UniPCM flow matching configuration.")
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")

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


ASPECT_RATIOS = {}

# Language-specific positive magic prompts for enhanced quality
POSITIVE_MAGIC = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图"
}



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Qwen-Image Lightning model and resources."""
        logging.basicConfig(level=logging.INFO)
        
        # Initialize LoRA tracking attributes
        self.loaded_loras = {}  # adapter_name -> (lora_source_id, lora_multiplier, created_names)
        
        # Determine which model variant to use
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        low_vram = variant.endswith("_low_vram")
        base_variant = variant[:-9] if low_vram else variant
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logging.info(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        # Lightning scheduler configuration
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        
        # Initialize scheduler with Lightning config
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        logging.info("Lightning scheduler configuration loaded")
        
        # Initialize the Qwen-Image pipeline
        if base_variant == "default":
            # Use the original Qwen-Image model with Lightning scheduler
            logging.info("Loading Qwen-Image model with Lightning scheduler")
            
            self.pipeline = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image",
                scheduler=scheduler,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            
        elif base_variant in GGUF_MODEL_VARIANTS:
            # Use GGUF quantized model from city96 with Lightning scheduler
            filename = GGUF_MODEL_VARIANTS[base_variant]
            gguf_repo = "city96/Qwen-Image-gguf"
            original_model_id = "Qwen/Qwen-Image"
            
            logging.info(f"Loading Qwen-Image GGUF variant: {base_variant} ({filename}) with Lightning scheduler")
            
            # Download the GGUF model file
            gguf_path = hf_hub_download(
                repo_id=gguf_repo,
                filename=filename,
                cache_dir="/tmp/qwen_image_cache"
            )
            logging.info(f"GGUF model downloaded to {gguf_path}")
            
            # Load GGUF transformer model
            logging.info("Loading GGUF quantized transformer...")
                            
            transformer = QwenImageTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.torch_dtype),
                torch_dtype=self.torch_dtype,
                config=original_model_id,
                subfolder="transformer",
            )
            logging.info("Successfully loaded GGUF transformer!")
            
            # Create pipeline with custom GGUF transformer and Lightning scheduler
            self.pipeline = QwenImagePipeline.from_pretrained(
                original_model_id,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=self.torch_dtype,
            )
                                
        else:
            logging.warning(f"Unknown variant '{variant}', falling back to original model with Lightning scheduler")
            
            self.pipeline = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image",
                scheduler=scheduler,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
        
        # Move to device first
        if not low_vram:
            self.pipeline = self.pipeline.to(self.device)
        
        # Load Lightning LoRA weights for acceleration
        try:
            print("🚀 Loading Lightning LoRA weights...")
            logging.info("Loading Lightning LoRA weights...")
            self.pipeline.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning", 
                weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
            )
            print("✅ Lightning LoRA weights loaded successfully")
            print("🔄 Fusing Lightning LoRA into model...")
            self.pipeline.fuse_lora()
            print("✅ Lightning LoRA fused into model - LoRA is now permanent part of model")
            logging.info("Successfully loaded and fused Lightning LoRA weights")
        except Exception as e:
            print(f"❌ Failed to load Lightning LoRA weights: {e}")
            logging.warning(f"Failed to load Lightning LoRA weights: {e}")
            logging.info("Continuing without Lightning LoRA weights")
        
        # Apply offloading strategy
        if low_vram:
            logging.info("Enabling leaf-level group offloading (low_vram mode)")
            onload_device = self.device
            offload_device = torch.device("cpu")

            self.pipeline.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            self.pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(self.pipeline.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")

            logging.info("Group offloading configured")
        else:
            # Default: move to device without offloading for better performance
            logging.info("Moving pipeline to device without offloading (default mode)")
            # Pipeline already moved to device earlier, ensure it stays there

        # Common optimizations
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()

        logging.info("Qwen-Image Lightning pipeline initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate image using Qwen-Image pipeline."""
        logging.info("Running in text-to-image generation mode")
        
        # Optionally switch to UniPC flow matching scheduler
        if input_data.use_unipcm_flow_matching:
            logging.info("Switching scheduler to UniPCM flow matching")
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
            )
        
        # Use provided width/height directly
        width, height = input_data.width, input_data.height
        logging.info(f"Using dimensions: {width}x{height}")
        
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
                    # Try deleting from transformer first
                    deleted = False
                    if hasattr(self.pipeline.transformer, 'delete_adapters'):
                        try:
                            self.pipeline.transformer.delete_adapters(created_name)
                            deleted = True
                            print(f"🗑️ Unloaded LoRA adapter '{created_name}' from transformer")
                        except Exception as e:
                            print(f"⚠️ Failed to delete LoRA from transformer: {e}, trying pipeline...")
                    
                    # Fallback to pipeline deletion
                    if not deleted and hasattr(self.pipeline, 'delete_adapters'):
                        try:
                            self.pipeline.delete_adapters(created_name)
                            deleted = True
                            print(f"🗑️ Unloaded LoRA adapter '{created_name}' from pipeline")
                        except Exception as e:
                            print(f"❌ Failed to delete LoRA from pipeline: {e}")
                    
                    if deleted:
                        logging.info(f"Unloaded previous LoRA adapter: {created_name}")
                    else:
                        print(f"⚠️ Could not delete adapter '{created_name}' - no delete_adapters method available")
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
                    self.pipeline,
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

        if active_adapters:
            print(f"🎯 Activating LoRA adapters: {active_adapters} with weights: {adapter_weights}")
            
            # For multiple adapters, activate each one individually
            if len(active_adapters) == 1:
                adapter_name = active_adapters[0]
                adapter_weight = adapter_weights[0]
                
                # Try activating on pipeline first (diffusers standard)
                activated = False
                if hasattr(self.pipeline, 'set_adapters'):
                    try:
                        self.pipeline.set_adapters(adapter_name, adapter_weights=adapter_weight)
                        print(f"✅ LoRA adapter '{adapter_name}' activated on pipeline with weight {adapter_weight}")
                        logging.info(f"Activated LoRA adapter on pipeline: {adapter_name} with weight: {adapter_weight}")
                        activated = True
                    except Exception as e:
                        print(f"⚠️ Failed to activate LoRA on pipeline: {e}, trying transformer...")
                
                # Fallback to transformer activation
                if not activated and hasattr(self.pipeline.transformer, 'set_adapters'):
                    try:
                        # PEFT doesn't support adapter_weights parameter
                        self.pipeline.transformer.set_adapters(adapter_name)
                        print(f"✅ LoRA adapter '{adapter_name}' activated on transformer (weight control not supported)")
                        logging.info(f"Activated LoRA adapter on transformer: {adapter_name}")
                        activated = True
                    except Exception as e:
                        print(f"❌ Failed to activate LoRA on transformer: {e}")
                
                if not activated:
                    print(f"⚠️ Found adapter '{adapter_name}' but neither pipeline nor transformer support set_adapters")
            else:
                # Multiple adapters - try list format on pipeline only
                activated = False
                if hasattr(self.pipeline, 'set_adapters'):
                    try:
                        self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
                        print(f"✅ Multiple LoRA adapters activated on pipeline successfully")
                        logging.info(f"Activated LoRA adapters on pipeline: {active_adapters} with weights: {adapter_weights}")
                        activated = True
                    except Exception as e:
                        print(f"❌ Failed to activate multiple LoRA adapters on pipeline: {e}")
                
                if not activated:
                    print(f"⚠️ Found {len(active_adapters)} adapters but pipeline doesn't support multiple adapter activation")
        else:
            print("ℹ️ No custom LoRA adapters to activate")

        # Configure first-block caching if requested
        # Disable any existing caches to avoid conflicts
        if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'disable_cache'):
            self.pipeline.transformer.disable_cache()
        if input_data.cache_threshold and input_data.cache_threshold > 0:
            logging.info(f"Enabling first-block cache with threshold={input_data.cache_threshold}")
            cache_cfg = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipeline.transformer.enable_cache(cache_cfg)
        
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

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logging.info("Resources cleaned up")