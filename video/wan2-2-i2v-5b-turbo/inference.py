import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import tempfile
from typing import Optional, List, Union, Tuple
from pydantic import BaseModel, Field
from PIL import Image
from diffusers import WanImageToVideoPipeline, ModularPipeline, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from diffusers.hooks import FirstBlockCacheConfig
# Remove hf_hub_download import since we don't use quantized models
from accelerate import Accelerator
from PIL import Image
import logging

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Turbo project only uses F16 model with turbo LoRA
DEFAULT_VARIANT = "default"

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_file: File = Field(description="LoRA weights file (.safetensors)")
    lora_multiplier: float = Field(default=1.0, ge=0.0, le=10.0, description="Multiplier for the LoRA effect")

class AppInput(BaseAppInput):
    image: File = Field(description="Input image for video generation")
    prompt: str = Field(description="Text prompt for video generation")
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
        description="Negative prompt to guide what to avoid in generation"
    )
    resolution: str = Field(default="720p", description="Resolution preset", enum=["480p", "720p"])
    num_frames: int = Field(default=121, description="Number of frames to generate")
    num_inference_steps: int = Field(default=4, description="Number of denoising steps")
    fps: int = Field(default=24, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    cache_threshold: float = Field(default=0.0, description="First Block Cache threshold (set 0 to disable)")
    end_image: Optional[File] = Field(default=None, description="Optional end image for first-to-last frame video generation")
    loras: Optional[List[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")

class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file")
    
def process_image(
    image: Union[str, Image.Image],
    max_area: int,
    mod_value: int,
) -> Image.Image:
    if isinstance(image, str):
        image = load_image(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Invalid image type: {type(image)}; only PIL Image or URL string supported")

    iw, ih = image.width, image.height
    dw, dh = mod_value, mod_value
    ratio = iw / ih
    ow = (max_area * ratio) ** 0.5
    oh = max_area / ow

    ow1 = int(ow // dw * dw)
    oh1 = int(max_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    oh2 = int(oh // dh * dh)
    ow2 = int(max_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        ow_final, oh_final = ow1, oh1
    else:
        ow_final, oh_final = ow2, oh2

    scale = max(ow_final / iw, oh_final / ih)
    resized = image.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

    x1 = (resized.width - ow_final) // 2
    y1 = (resized.height - oh_final) // 2
    cropped = resized.crop((x1, y1, x1 + ow_final, y1 + oh_final))
    return cropped

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2-TI2V-5B Image-to-Video pipeline and resources here."""
        print("Setting up Wan2.2-TI2V-5B Image-to-Video pipeline...")

        # Store resolution defaults (using TI2V resolution standards)
        self.resolution_presets = {
            "480p": {"max_area": 480 * 832},
            "720p": {"max_area": 704 * 1280}  # Updated to TI2V's 704 height
        }
        # Initialize accelerator
        self.accelerator = Accelerator()

        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")

        print(f"Metadata: {metadata}")
        # Get variant and determine if using quantization
        if isinstance(metadata, dict) and 'app_variant' in metadata:
            variant = metadata['app_variant']
        elif hasattr(metadata, 'app_variant'):
            variant = metadata.app_variant
        else:
            variant = DEFAULT_VARIANT

        original_variant = variant  # Preserve original for special variant handling

        print(f"Loading model variant: {variant}")

        # Model ID for the TI2V 5B variant (using I2V pipeline with TI2V model)
        self.model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

        self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)

        # Initialize LoRA tracking
        self.loaded_loras = {}  # adapter_name -> (source, multiplier, created_names)

        # Load standard F16 pipeline (turbo project only uses F16)
        print("Loading F16 Wan2.2-TI2V-5B I2V pipeline for turbo mode...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            vae=self.vae,
        )

        # Apply device placement based on variant
        if original_variant == "default":
            # Default variant (turbo): load everything directly to device without offloading
            print("🚀 Turbo mode: loading all components directly to device without offloading")
            self.pipe.to(self.device)
        elif original_variant == "turbo_offload":
            # Turbo with offloading: use CPU offload for memory efficiency
            print("🚀 Turbo with offloading: enabling CPU offload")
            self.pipe.enable_model_cpu_offload()
        else:
            # Fallback: enable CPU offload for memory efficiency
            print("🚀 Fallback: enabling CPU offload")
            self.pipe.enable_model_cpu_offload()

        # Set default scheduler for normal mode
        self.default_scheduler = self.pipe.scheduler

        # Auto-load Turbo LoRA (always load in turbo project)
        print(f"🚀 Turbo project: loading WAN22 Turbo LoRA for variant '{original_variant}'...")
        try:
            self.pipe.load_lora_weights(
                "Kijai/WanVideo_comfy",
                weight_name="LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors",
                adapter_name="turbo_lora",
            )
            self.pipe.set_adapters("turbo_lora", adapter_weights=1.0)
            print("✅ WAN22 Turbo LoRA loaded and activated successfully")

            # Fuse LoRA weights into the base model for better performance
            print("🔗 Fusing Turbo LoRA weights into base model...")
            self.pipe.fuse_lora(adapter_names=["turbo_lora"])
            print("✅ Turbo LoRA weights fused into base model")

            # Unload LoRA weights to save memory after fusion
            print("🗑️ Unloading Turbo LoRA weights after fusion...")
            self.pipe.unload_lora_weights()
            print("✅ Turbo LoRA weights unloaded, fused weights retained in base model")

            self.turbo_lora_enabled = True
        except Exception as e:
            print(f"⚠️ Failed to load/fuse Turbo LoRA: {e}")
            self.turbo_lora_enabled = False

        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from image and text prompt."""
        print(f"Generating video with prompt: {input_data.prompt}")

        # Use resolution preset
        preset = self.resolution_presets.get(input_data.resolution, self.resolution_presets["720p"])
        max_area = preset["max_area"]
        print(f"Using resolution: {input_data.resolution}, max area: {max_area}")

        # Handle LoRA adapters
        loras = getattr(input_data, "loras", None) or []
        requested_by_name = {l.adapter_name: l for l in loras}

        # Unload adapters that changed or are no longer needed
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
                    if hasattr(self.pipe.transformer, 'delete_adapters'):
                        try:
                            self.pipe.transformer.delete_adapters(created_name)
                            deleted = True
                            print(f"🗑️ Unloaded LoRA adapter '{created_name}' from transformer")
                        except Exception as e:
                            print(f"⚠️ Failed to delete LoRA from transformer: {e}, trying pipeline...")

                    # Fallback to pipeline deletion
                    if not deleted and hasattr(self.pipe, 'delete_adapters'):
                        try:
                            self.pipe.delete_adapters(created_name)
                            deleted = True
                            print(f"🗑️ Unloaded LoRA adapter '{created_name}' from pipeline")
                        except Exception as e:
                            print(f"❌ Failed to delete LoRA from pipeline: {e}")

                    if deleted:
                        logging.info(f"Unloaded previous LoRA adapter: {created_name}")
                del self.loaded_loras[adapter_name]

        # Load new/changed adapters
        for lora in loras:
            needs_load = (
                lora.adapter_name not in self.loaded_loras
                or self.loaded_loras[lora.adapter_name][0] != lora.lora_file.path
                or self.loaded_loras[lora.adapter_name][1] != lora.lora_multiplier
            )
            if needs_load:
                created_names = self._load_lora_adapter(
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

        # Apply adapter activation with fallback strategy
        if active_adapters:
            print(f"🎯 Activating LoRA adapters: {active_adapters} with weights: {adapter_weights}")

            # Handle single adapter activation (diffusers standard API)
            if len(active_adapters) == 1:
                adapter_name = active_adapters[0]
                adapter_weight = adapter_weights[0]

                # Try activating on pipeline first (diffusers standard)
                activated = False
                if hasattr(self.pipe, 'set_adapters'):
                    try:
                        self.pipe.set_adapters(adapter_name, adapter_weights=adapter_weight)
                        print(f"✅ LoRA adapter '{adapter_name}' activated on pipeline successfully")
                        activated = True
                    except Exception as e:
                        print(f"⚠️ Failed to activate LoRA on pipeline: {e}, trying transformer...")

                # Fallback to transformer activation
                if not activated and hasattr(self.pipe.transformer, 'set_adapters'):
                    try:
                        self.pipe.transformer.set_adapters(adapter_name, adapter_weights=adapter_weight)
                        print(f"✅ LoRA adapter '{adapter_name}' activated on transformer successfully")
                        activated = True
                    except Exception as e:
                        print(f"❌ Failed to activate LoRA on transformer: {e}")

                if not activated:
                    print(f"⚠️ Found adapter '{adapter_name}' but neither pipeline nor transformer support set_adapters")
        else:
            print("ℹ️ No custom LoRA adapters to activate")

        # Configure scheduler based on turbo variant (always turbo mode in this project)
        turbo_mode = self.turbo_lora_enabled
        if turbo_mode:
            print("🚀 Enabling Turbo mode with optimized scheduler...")
            # Use UniPCM scheduler with flow_shift for faster generation
            #self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            #    self.pipe.scheduler.config, flow_shift=8.0
            #)
            # Use the requested steps (default is 4 for turbo)
            effective_steps = input_data.num_inference_steps
            print(f"Turbo mode: using {effective_steps}")# inference steps with UniPCM scheduler")
        else:
            # Use default scheduler
            self.pipe.scheduler = self.default_scheduler
            effective_steps = input_data.num_inference_steps

        # 5B model uses a single transformer; do not set boundary ratio or any dual-transformer config

        # Load and process input image
        image = Image.open(input_data.image.path).convert("RGB")
        print(f"Loaded image: {image.size}")

        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]

        # Resize image according to pipeline requirements
        resized_image = process_image(
            image=input_data.image.path,
            max_area=max_area,
            mod_value=mod_value,
        )

        last_image = None
        if input_data.end_image is not None:
            last_image = process_image(
                image=input_data.end_image.path,
                max_area=max_area,
                mod_value=mod_value,
            )

        width = resized_image.width
        height = resized_image.height

        # Set seed if provided
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
            print(f"Using seed: {input_data.seed}")
        

      # Configure caching if thresholds are non-zero
        # First disable any existing caching to prevent conflicts
        if hasattr(self.pipe.transformer, 'disable_cache'):
            self.pipe.transformer.disable_cache()
        
        if input_data.cache_threshold > 0:
            print(f"Enabling cache for transformer with threshold: {input_data.cache_threshold}")
            cache_config = FirstBlockCacheConfig(threshold=input_data.cache_threshold)
            self.pipe.transformer.enable_cache(cache_config)
        
        # Generate video
        print("Starting video generation...")
        output = self.pipe(
            image=resized_image,
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            height=height,
            width=width,
            num_frames=input_data.num_frames,
            guidance_scale=1.0,  # Use default guidance scale for turbo mode
            num_inference_steps=effective_steps,
            generator=generator,
            last_image=last_image,
        ).frames[0]
        
        print("Video generation complete, exporting...")
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Export video
        export_to_video(output, output_path, fps=input_data.fps)
        
        print(f"Video exported to: {output_path}")
        
        return AppOutput(file=File(path=output_path))

    def _load_lora_adapter(self, lora_source, adapter_name="lora", lora_multiplier=1.0):
        """
        Load LoRA adapter with fallback strategy: transformer first, then pipeline.
        Supports: local files, HuggingFace URLs, direct URLs.
        """
        if not lora_source:
            return []

        def _load_single_lora(load_kwargs, base_adapter_name: str):
            # Try loading on transformer first (more direct LoRA support)
            try:
                transformer = getattr(self.pipe, 'transformer', None)
                if transformer and hasattr(transformer, 'load_lora_weights'):
                    print(f"🔄 Loading LoRA adapter '{base_adapter_name}' onto transformer...")
                    load_kwargs["adapter_name"] = base_adapter_name
                    transformer.load_lora_weights(**load_kwargs)
                    print(f"✅ LoRA adapter '{base_adapter_name}' loaded successfully onto transformer")
                    logging.info(f"Loaded LoRA adapter {base_adapter_name} onto transformer")
                    return base_adapter_name
            except Exception as e:
                print(f"⚠️ Failed to load LoRA on transformer: {e}, trying pipeline...")
                logging.warning(f"Failed to load LoRA {base_adapter_name} on transformer: {e}")

            # Fallback to pipeline (diffusers standard approach)
            try:
                if hasattr(self.pipe, 'load_lora_weights'):
                    print(f"🔄 Loading LoRA adapter '{base_adapter_name}' onto pipeline...")
                    load_kwargs["adapter_name"] = base_adapter_name
                    self.pipe.load_lora_weights(**load_kwargs)
                    print(f"✅ LoRA adapter '{base_adapter_name}' loaded successfully onto pipeline")
                    logging.info(f"Loaded LoRA adapter {base_adapter_name} onto pipeline")
                    return base_adapter_name
                else:
                    print(f"❌ Neither transformer nor pipeline support load_lora_weights method")
            except Exception as e:
                print(f"❌ Failed to load LoRA '{base_adapter_name}' on pipeline: {e}")
                logging.error(f"Failed to load LoRA {base_adapter_name}: {e}")
            return None

        # Handle different source types:

        # 1. Local file path
        if isinstance(lora_source, str) and os.path.isfile(lora_source):
            load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_source}
            created = _load_single_lora(load_kwargs, adapter_name)
            return [created] if created else []

        # 2. HuggingFace blob URL (https://huggingface.co/user/repo/blob/main/file.safetensors)
        if isinstance(lora_source, str) and "huggingface.co" in lora_source and "/blob/" in lora_source:
            parts = lora_source.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'blob' in parts:
                repo_start = parts.index('huggingface.co') + 1
                blob_index = parts.index('blob')
                repo_id = '/'.join(parts[repo_start:blob_index])
                weight_name = '/'.join(parts[blob_index + 2:])
                load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
                created = _load_single_lora(load_kwargs, adapter_name)
                return [created] if created else []

        # 3. HuggingFace resolve URL (https://huggingface.co/user/repo/resolve/main/file.safetensors)
        elif isinstance(lora_source, str) and "huggingface.co" in lora_source and "/resolve/" in lora_source:
            parts = lora_source.split('/')
            if len(parts) >= 7 and 'huggingface.co' in parts and 'resolve' in parts:
                repo_start = parts.index('huggingface.co') + 1
                resolve_index = parts.index('resolve')
                repo_id = '/'.join(parts[repo_start:resolve_index])
                weight_name = '/'.join(parts[resolve_index + 2:])
                load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
                created = _load_single_lora(load_kwargs, adapter_name)
                return [created] if created else []

        # 4. HuggingFace repository (user/repo or user/repo/file.safetensors)
        elif isinstance(lora_source, str) and "/" in lora_source and not lora_source.startswith('http'):
            parts = lora_source.split('/')
            if len(parts) == 2:  # Just repo
                load_kwargs = {"repo_id": lora_source}
                created = _load_single_lora(load_kwargs, adapter_name)
                return [created] if created else []
            elif len(parts) > 2:  # Repo with file path
                repo_id = '/'.join(parts[:2])
                weight_name = '/'.join(parts[2:])
                load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
                created = _load_single_lora(load_kwargs, adapter_name)
                return [created] if created else []

        else:
            raise ValueError(f"Unsupported LoRA source format: {lora_source}")

    async def unload(self):
        """Clean up resources here."""
        print("Cleaning up...")
        if hasattr(self, 'pipe'):
            del self.pipe
        
        # Clear GPU cache if using CUDA
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        print("Cleanup complete!") 