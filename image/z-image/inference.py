import os
import sys
import re
import random
import warnings
import logging

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Environment setup - must be done before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from pydantic import Field
from typing import Optional, Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Resolution options
RESOLUTION_CHOICES = [
    "1024x1024",
    "1152x896",
    "896x1152",
    "1152x864",
    "864x1152",
    "1248x832",
    "832x1248",
    "1280x720",
    "720x1280",
    "1344x576",
    "576x1344",
    "1280x1280",
    "1440x1120",
    "1120x1440",
    "1472x1104",
    "1104x1472",
    "1536x1024",
    "1024x1536",
    "1600x896",
    "896x1600",
    "1680x720",
    "720x1680",
]


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt describing the desired image content")
    resolution: str = Field(
        default="1024x1024",
        description="Output resolution in WIDTHxHEIGHT format (e.g., '1024x1024', '1280x720')"
    )
    seed: int = Field(
        default=-1,
        description="Seed for reproducible generation. Use -1 for random seed"
    )
    steps: int = Field(
        default=9,
        description="Number of inference steps for the diffusion process (recommended: 8-10 for turbo)"
    )
    shift: float = Field(
        default=3.0,
        description="Time shift parameter for the flow matching scheduler (1.0-10.0)"
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image file")
    seed: int = Field(description="The seed used for generation")
    resolution: str = Field(description="The resolution of the generated image")


def get_resolution(resolution: str) -> tuple[int, int]:
    """Parse resolution string to width and height."""
    match = re.search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.pipe = None
        self.model_path = "Tongyi-MAI/Z-Image-Turbo"
        self.attention_backend = "flash_3"
        self.enable_compile = False  # Disable compile for faster cold start

    async def setup(self, metadata):
        """Initialize the Z-Image model and pipeline."""
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers import ZImagePipeline
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
        from transformers import AutoModel, AutoTokenizer
        from accelerate import Accelerator

        # Get device
        accelerator = Accelerator()
        self.device = accelerator.device
        
        # Check variant for potential low-vram mode
        variant = getattr(metadata, "app_variant", "default")
        
        if variant == "low_vram":
            self.enable_compile = False
            self.attention_backend = "native"
        
        print(f"Loading Z-Image model from {self.model_path}...")
        print(f"Device: {self.device}, Variant: {variant}")
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map=str(self.device),
        )
        
        # Load text encoder
        text_encoder = AutoModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map=str(self.device),
        ).eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            subfolder="tokenizer"
        )
        tokenizer.padding_side = "left"
        
        # Configure torch compile if enabled
        if self.enable_compile:
            print("Enabling torch.compile optimizations...")
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
            torch._inductor.config.max_autotune_gemm = True
            torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
            torch._inductor.config.triton.cudagraphs = False
        
        # Create pipeline without transformer initially
        self.pipe = ZImagePipeline(
            scheduler=None,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=None
        )
        
        if self.enable_compile:
            self.pipe.vae.disable_tiling()
        
        # Load transformer
        transformer = ZImageTransformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer"
        ).to(self.device, torch.bfloat16)
        
        self.pipe.transformer = transformer
        self.pipe.transformer.set_attention_backend(self.attention_backend)
        
        # Compile transformer if enabled
        if self.enable_compile:
            print("Compiling transformer...")
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False
            )
        
        self.pipe.to(self.device, torch.bfloat16)
        print("Z-Image model loaded successfully!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image using the Z-Image model."""
        from diffusers import FlowMatchEulerDiscreteScheduler
        import tempfile
        
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Please ensure setup() completed successfully.")
        
        # Parse resolution
        width, height = get_resolution(input_data.resolution)
        
        # Handle seed
        if input_data.seed == -1:
            seed = random.randint(1, 2**32 - 1)
        else:
            seed = input_data.seed
        
        # Validate steps
        steps = max(1, min(100, input_data.steps))
        
        # Validate shift
        shift = max(1.0, min(10.0, input_data.shift))
        
        print(f"Generating image: prompt='{input_data.prompt[:50]}...', "
              f"resolution={width}x{height}, seed={seed}, steps={steps}, shift={shift}")
        
        # Create generator with seed
        generator = torch.Generator(device=str(self.device)).manual_seed(seed)
        
        # Create scheduler with shift parameter
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=shift
        )
        self.pipe.scheduler = scheduler
        
        # Generate image
        # Note: Z-Image Turbo works best with guidance_scale=0.0
        result = self.pipe(
            prompt=input_data.prompt,
            height=height,
            width=width,
            guidance_scale=0.0,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
        )
        
        image = result.images[0]
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            output_path = tmp.name
        
        image.save(output_path, format="PNG")
        
        return AppOutput(
            image=File(path=output_path),
            seed=seed,
            resolution=f"{width}x{height}"
        )

    async def unload(self):
        """Clean up resources."""
        import gc
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("Z-Image model unloaded.")
