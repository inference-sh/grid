import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import torch
from diffusers.pipelines import HiDreamImagePipeline
from diffusers import HiDreamImageTransformer2DModel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze, qfloat8, quantize
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from diffusers import GGUFQuantizationConfig
import logging


MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"

# GGUF repository IDs
GGUF_REPOS = {
    "full": "city96/HiDream-I1-Full-gguf",
    "dev": "city96/HiDream-I1-Dev-gguf", 
    "fast": "city96/HiDream-I1-Fast-gguf"
}

# Scheduler mapping
SCHEDULER_MAPPING = {
    "flow_unipc": FlowUniPCMultistepScheduler,
    "flash_flow": FlashFlowMatchEulerDiscreteScheduler
}

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 1.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler,  # Normal scheduler
        "type": "regular"
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler,  # Normal scheduler  
        "type": "regular"
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 1.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler,  # Normal scheduler
        "type": "regular"
    },
    "fp8-full": {
        "path": "https://huggingface.co/boricuapab/hidream_e1_full_bf16-fp8/hidream_e1_full_bf16-fp8.safetensors",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler,
        "type": "fp8"
    },
    # Full GGUF variants (Steps-50, Shift-3.0, CFG-5, Scheduler-Normal)
    "full-f16": {
        "repo": "full", "filename": "hidream-i1-full-F16.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q8": {
        "repo": "full", "filename": "hidream-i1-full-Q8_0.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q6k": {
        "repo": "full", "filename": "hidream-i1-full-Q6_K.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q5km": {
        "repo": "full", "filename": "hidream-i1-full-Q5_K_M.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q5ks": {
        "repo": "full", "filename": "hidream-i1-full-Q5_K_S.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q51": {
        "repo": "full", "filename": "hidream-i1-full-Q5_1.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q50": {
        "repo": "full", "filename": "hidream-i1-full-Q5_0.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q4km": {
        "repo": "full", "filename": "hidream-i1-full-Q4_K_M.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q4ks": {
        "repo": "full", "filename": "hidream-i1-full-Q4_K_S.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q41": {
        "repo": "full", "filename": "hidream-i1-full-Q4_1.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q40": {
        "repo": "full", "filename": "hidream-i1-full-Q4_0.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q3km": {
        "repo": "full", "filename": "hidream-i1-full-Q3_K_M.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q3ks": {
        "repo": "full", "filename": "hidream-i1-full-Q3_K_S.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },
    "full-q2k": {
        "repo": "full", "filename": "hidream-i1-full-Q2_K.gguf",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler, "type": "gguf"
    },

    # Dev GGUF variants (Steps-28, Shift-6.0, CFG-1, Scheduler-Normal)
    "dev-f16": {
        "repo": "dev", "filename": "hidream-i1-dev-F16.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q8": {
        "repo": "dev", "filename": "hidream-i1-dev-Q8_0.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q6k": {
        "repo": "dev", "filename": "hidream-i1-dev-Q6_K.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q5km": {
        "repo": "dev", "filename": "hidream-i1-dev-Q5_K_M.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q5ks": {
        "repo": "dev", "filename": "hidream-i1-dev-Q5_K_S.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q51": {
        "repo": "dev", "filename": "hidream-i1-dev-Q5_1.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q50": {
        "repo": "dev", "filename": "hidream-i1-dev-Q5_0.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q4km": {
        "repo": "dev", "filename": "hidream-i1-dev-Q4_K_M.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q4ks": {
        "repo": "dev", "filename": "hidream-i1-dev-Q4_K_S.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q41": {
        "repo": "dev", "filename": "hidream-i1-dev-Q4_1.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q40": {
        "repo": "dev", "filename": "hidream-i1-dev-Q4_0.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q3km": {
        "repo": "dev", "filename": "hidream-i1-dev-Q3_K_M.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q3ks": {
        "repo": "dev", "filename": "hidream-i1-dev-Q3_K_S.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "dev-q2k": {
        "repo": "dev", "filename": "hidream-i1-dev-Q2_K.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },

    # Fast GGUF variants (Steps-16, Shift-3.0, CFG-1, Scheduler-Normal)
    "fast-f16": {
        "repo": "fast", "filename": "hidream-i1-fast-F16.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q8": {
        "repo": "fast", "filename": "hidream-i1-fast-Q8_0.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q6k": {
        "repo": "fast", "filename": "hidream-i1-fast-Q6_K.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q5km": {
        "repo": "fast", "filename": "hidream-i1-fast-Q5_K_M.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q5ks": {
        "repo": "fast", "filename": "hidream-i1-fast-Q5_K_S.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q51": {
        "repo": "fast", "filename": "hidream-i1-fast-Q5_1.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q50": {
        "repo": "fast", "filename": "hidream-i1-fast-Q5_0.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q4km": {
        "repo": "fast", "filename": "hidream-i1-fast-Q4_K_M.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q4ks": {
        "repo": "fast", "filename": "hidream-i1-fast-Q4_K_S.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q41": {
        "repo": "fast", "filename": "hidream-i1-fast-Q4_1.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q40": {
        "repo": "fast", "filename": "hidream-i1-fast-Q4_0.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q3km": {
        "repo": "fast", "filename": "hidream-i1-fast-Q3_K_M.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q3ks": {
        "repo": "fast", "filename": "hidream-i1-fast-Q3_K_S.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
    "fast-q2k": {
        "repo": "fast", "filename": "hidream-i1-fast-Q2_K.gguf",
        "guidance_scale": 1.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler, "type": "gguf"
    },
}

MAX_TOTAL_PIXELS = 1920 * 1080  # Maximum total pixels allowed

def round_to_multiple_of_8(n: int) -> int:
    """Round a number to the nearest multiple of 8."""
    return ((n + 4) // 8) * 8

def adjust_dimensions(width: int, height: int) -> tuple[int, int]:
    """Adjust dimensions to be multiples of 8 and respect max pixel count."""
    # First round to multiples of 8
    width = round_to_multiple_of_8(width)
    height = round_to_multiple_of_8(height)
    
    # If total pixels exceed max, scale down while maintaining aspect ratio
    total_pixels = width * height
    if total_pixels > MAX_TOTAL_PIXELS:
        aspect_ratio = width / height
        # Solve: w * h = MAX_TOTAL_PIXELS and w/h = aspect_ratio
        new_height = int((MAX_TOTAL_PIXELS / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        # Round to multiples of 8 again
        width = round_to_multiple_of_8(new_width)
        height = round_to_multiple_of_8(new_height)
    
    return width, height

def get_scheduler_class(scheduler_name: str, default_scheduler):
    """Get scheduler class from name or return default."""
    if scheduler_name is None:
        return default_scheduler
    return SCHEDULER_MAPPING.get(scheduler_name, default_scheduler)

class AppInput(BaseAppInput):
    prompt: str = Field(description="The prompt to generate an image from")
    width: int = Field(default=1024, ge=8, description="The width of the generated image (will be adjusted to nearest multiple of 8)")
    height: int = Field(default=1024, ge=8, description="The height of the generated image (will be adjusted to nearest multiple of 8)")
    seed: Optional[int] = Field(default=-1, description="The seed for the random number generator (-1 for random)")
    
    # Generation parameters
    num_inference_steps: int = Field(
        default=None, 
        ge=1, 
        le=100, 
        description="Number of denoising steps (uses variant default if not provided)"
    )
    guidance_scale: float = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="CFG scale - how closely to follow the prompt (uses variant default if not provided)"
    )
    shift: float = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Shift parameter for scheduler (uses variant default if not provided)"
    )
    scheduler_type: str = Field(
        default=None,
        description="Scheduler type: 'flow_unipc' or 'flash_flow' (uses variant default if not provided)"
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.accelerator = Accelerator()
        print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"[DEBUG] Accelerator device: {self.accelerator.device}")
        
        # Set up variant and model type
        self.variant = getattr(metadata, "app_variant", "default")
        
        # Map variants to model configs
        variant_mapping = {
            "default": "fast",
            "dev": "dev", 
            "full": "full",
            "fp8-full": "fp8-full",
            # Full GGUF variants
            "full-f16": "full-f16", "full-q8": "full-q8", "full-q6k": "full-q6k",
            "full-q5km": "full-q5km", "full-q5ks": "full-q5ks", "full-q51": "full-q51", "full-q50": "full-q50",
            "full-q4km": "full-q4km", "full-q4ks": "full-q4ks", "full-q41": "full-q41", "full-q40": "full-q40",
            "full-q3km": "full-q3km", "full-q3ks": "full-q3ks", "full-q2k": "full-q2k",
            # Dev GGUF variants  
            "dev-f16": "dev-f16", "dev-q8": "dev-q8", "dev-q6k": "dev-q6k",
            "dev-q5km": "dev-q5km", "dev-q5ks": "dev-q5ks", "dev-q51": "dev-q51", "dev-q50": "dev-q50",
            "dev-q4km": "dev-q4km", "dev-q4ks": "dev-q4ks", "dev-q41": "dev-q41", "dev-q40": "dev-q40",
            "dev-q3km": "dev-q3km", "dev-q3ks": "dev-q3ks", "dev-q2k": "dev-q2k",
            # Fast GGUF variants
            "fast-f16": "fast-f16", "fast-q8": "fast-q8", "fast-q6k": "fast-q6k",
            "fast-q5km": "fast-q5km", "fast-q5ks": "fast-q5ks", "fast-q51": "fast-q51", "fast-q50": "fast-q50",
            "fast-q4km": "fast-q4km", "fast-q4ks": "fast-q4ks", "fast-q41": "fast-q41", "fast-q40": "fast-q40",
            "fast-q3km": "fast-q3km", "fast-q3ks": "fast-q3ks", "fast-q2k": "fast-q2k"
        }
        
        self.model = variant_mapping.get(self.variant, "fast")
        self.config = MODEL_CONFIGS[self.model]
        
        print(f"[DEBUG] Using model variant: {self.variant} -> {self.model}")
        print(f"[DEBUG] Model type: {self.config['type']}")
        
        # Store config for later use in run method
        self.model_config = self.config
        
        # Load tokenizer and text encoder
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            LLAMA_MODEL_NAME,
            use_fast=False
        )
        
        self.text_encoder = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16
        )

        # Load and set up the pipeline based on model type
        if self.config["type"] == "gguf":
            # GGUF loading following official Diffusers documentation pattern
            repo_key = self.config["repo"]
            repo_id = GGUF_REPOS[repo_key]
            filename = self.config["filename"]
            print(f"[DEBUG] Downloading {filename} from {repo_id}...")
            
            # Download GGUF file
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"[DEBUG] Model downloaded to {ckpt_path}")
            
            # Load transformer from GGUF file with proper quantization
            transformer = HiDreamImageTransformer2DModel.from_single_file(
                ckpt_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16
            )
            print(f"[DEBUG] GGUF transformer loaded")

            # Create pipeline with GGUF transformer directly (avoid loading original transformer)
            self.pipe = HiDreamImagePipeline.from_pretrained(
                "HiDream-ai/HiDream-I1-Full",
                transformer=transformer,  # Use GGUF transformer directly
                tokenizer_4=self.tokenizer,
                text_encoder_4=self.text_encoder,
                torch_dtype=torch.bfloat16
            )
            self.pipe.enable_model_cpu_offload()
            print(f"[DEBUG] Pipeline created with GGUF transformer")
            
        elif self.config["type"] == "fp8":
            # FP8 quantization loading
            print(f"[DEBUG] Loading FP8 model from: {self.config['path']}")
            
            # First load the transformer separately
            transformer = HiDreamImageTransformer2DModel.from_single_file(
                self.config["path"],
                torch_dtype=torch.bfloat16
            )
            # Quantize and freeze the transformer
            quantize(transformer, weights=qfloat8)
            freeze(transformer)
            transformer.to(self.accelerator.device)

            # Load the rest of the pipeline without transformer
            self.pipe = HiDreamImagePipeline.from_pretrained(
                "HiDream-ai/HiDream-I1-Full",
                tokenizer_4=self.tokenizer,
                text_encoder_4=self.text_encoder,
                transformer=None,  # We'll set this after
                torch_dtype=torch.bfloat16
            )
            # Set the quantized transformer
            self.pipe.transformer = transformer
            
            # Quantize text encoder if needed
            quantize(self.pipe.text_encoder_4, weights=qfloat8)
            freeze(self.pipe.text_encoder_4)

            self.pipe.enable_model_cpu_offload()
        else:
            # Regular model loading
            print(f"[DEBUG] Loading regular model from: {self.config['path']}")
            self.pipe = HiDreamImagePipeline.from_pretrained(
                self.config["path"],
                tokenizer_4=self.tokenizer,
                text_encoder_4=self.text_encoder,
                torch_dtype=torch.bfloat16
            )
            self.pipe.to(self.accelerator.device)
            self.pipe.enable_model_cpu_offload()

        print(f"[DEBUG] Pipeline loaded and moved to device: {self.accelerator.device}")
        self.device = self.accelerator.device

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Get configuration for current model        
        # Adjust dimensions to be multiples of 8 and respect max pixel count
        width, height = adjust_dimensions(input_data.width, input_data.height)
        
        # Set up generator for reproducibility
        seed = input_data.seed if input_data.seed is not None and input_data.seed != -1 else torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        # Get user parameters or use variant defaults
        num_inference_steps = input_data.num_inference_steps if input_data.num_inference_steps is not None else self.model_config["num_inference_steps"]
        guidance_scale = input_data.guidance_scale if input_data.guidance_scale is not None else self.model_config["guidance_scale"]
        shift = input_data.shift if input_data.shift is not None else self.model_config["shift"]
        
        # Create scheduler with user parameters
        scheduler_class = get_scheduler_class(input_data.scheduler_type, self.model_config["scheduler"])
        scheduler = scheduler_class(
            num_train_timesteps=1000,
            shift=shift,
            use_dynamic_shifting=False
        )
        
        print(f"[DEBUG] Using parameters: steps={num_inference_steps}, cfg={guidance_scale}, shift={shift}, scheduler={scheduler_class.__name__}")

        # Generate image
        images = self.pipe(
            input_data.prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            scheduler=scheduler
        ).images

        # Save the generated image
        output_path = "/tmp/generated_image.png"
        images[0].save(output_path)
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        del self.pipe
        del self.text_encoder
        del self.tokenizer
        torch.cuda.empty_cache()