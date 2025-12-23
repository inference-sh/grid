import os
import sys
import random
import warnings
import logging
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Environment setup - must be done before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Maximum total pixels allowed (1920x1080)
MAX_TOTAL_PIXELS = 1920 * 1080


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


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt describing the desired image content")
    width: int = Field(
        default=1024,
        description="Output image width in pixels (will be rounded to multiple of 8)",
    )
    height: int = Field(
        default=1024,
        description="Output image height in pixels (will be rounded to multiple of 8)",
    )
    seed: int = Field(
        default=-1,
        description="Seed for reproducible generation. Use -1 for random seed",
    )
    steps: int = Field(
        default=9,
        description="Number of inference steps for the diffusion process (recommended: 8-10 for turbo)",
    )
    shift: float = Field(
        default=3.0,
        description="Time shift parameter for the flow matching scheduler (1.0-10.0)",
    )

class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Z-Image model and pipeline."""

        # Get device
        accelerator = Accelerator()
        self.device = accelerator.device

        print(f"Device: {self.device}")
        # Create pipeline without transformer initially
        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.pipe.to(device=self.device, dtype=torch.bfloat16)


        # [Optional] Attention Backend
        # Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
        self.pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
        # pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

        # [Optional] Model Compilation
        # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
        # pipe.transformer.compile()

        # [Optional] CPU Offloading
        # Enable CPU offloading for memory-constrained devices.
        # pipe.enable_model_cpu_offload()


        print("Z-Image model loaded successfully!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image using the Z-Image model."""
        from diffusers import FlowMatchEulerDiscreteScheduler
        import tempfile

        if self.pipe is None:
            raise RuntimeError(
                "Model not loaded. Please ensure setup() completed successfully."
            )

        # Adjust dimensions to be multiples of 8 and respect max pixel count
        width, height = adjust_dimensions(input_data.width, input_data.height)

        # Handle seed
        if input_data.seed == -1:
            seed = random.randint(1, 2**32 - 1)
        else:
            seed = input_data.seed

        # Validate steps
        steps = max(1, min(100, input_data.steps))

        # Validate shift
        shift = max(1.0, min(10.0, input_data.shift))

        print(
            f"Generating image: prompt='{input_data.prompt[:50]}...', "
            f"size={width}x{height}, seed={seed}, steps={steps}, shift={shift}"
        )

        # Create generator with seed
        generator = torch.Generator(device=str(self.device)).manual_seed(seed)

        # Create scheduler with shift parameter
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift
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
            image=File(path=output_path)
        )

