from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
import torch
from diffusers.pipelines import HiDreamImagePipeline
from diffusers.models import HiDreamImageTransformer2DModel
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlowMatchEulerDiscreteScheduler
    }
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

class AppInput(BaseAppInput):
    prompt: str = Field(description="The prompt to generate an image from")
    width: int = Field(default=1024, ge=8, description="The width of the generated image (will be adjusted to nearest multiple of 8)")
    height: int = Field(default=1024, ge=8, description="The height of the generated image (will be adjusted to nearest multiple of 8)")
    seed: int = Field(default=-1, ge=-1, le=1000000, description="The seed for the random number generator")

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    async def setup(self, metadata: dict):
        """Initialize your model and resources here."""
        self.model_type = "full"  # Default model type
        config = MODEL_CONFIGS[self.model_type]
        
        # Model configuration
        self.scheduler = config["scheduler"](
            num_train_timesteps=1000,
            shift=config["shift"],
            use_dynamic_shifting=False
        )
        
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

        self.pipe = HiDreamImagePipeline.from_pretrained(
            config["path"],
            scheduler=self.scheduler,
            tokenizer_4=self.tokenizer,
            text_encoder_4=self.text_encoder,
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()
        
        self.variant = metadata.app_variant
        self.model = "fast"
        if self.variant == "default":
            self.model = "fast"
        elif self.variant == "full":
            self.model = "full"
        elif self.variant == "dev":
            self.model = "dev"
        self.config = MODEL_CONFIGS[self.model]

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Get configuration for current model        
        # Adjust dimensions to be multiples of 8 and respect max pixel count
        width, height = adjust_dimensions(input_data.width, input_data.height)
        
        # Set up generator for reproducibility
        seed = input_data.seed if input_data.seed != -1 else torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate image
        images = self.pipe(
            input_data.prompt,
            height=height,
            width=width,
            guidance_scale=self.config["guidance_scale"],
            num_inference_steps=self.config["num_inference_steps"],
            num_images_per_prompt=1,
            generator=generator
        ).images

        # Save the generated image
        output_path = "/tmp/generated_image.png"
        images[0].save(output_path)
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        del self.pipe
        del self.transformer
        del self.text_encoder
        del self.tokenizer
        torch.cuda.empty_cache()