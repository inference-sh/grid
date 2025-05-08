from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
import torch
from diffusers.pipelines import HiDreamImagePipeline
from diffusers.models import HiDreamImageTransformer2DModel
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

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
        "scheduler": UniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlowMatchEulerDiscreteScheduler
    }
}

RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

class AppInput(BaseAppInput):
    prompt: str = Field(description="The prompt to generate an image from")
    resolution: str = Field(default="1024 × 1024 (Square)", enum=RESOLUTION_OPTIONS, description="The resolution of the generated image")
    seed: int = Field(default=-1, ge=-1, le=1000000, description="The seed for the random number generator")

class AppOutput(BaseAppOutput):
    result: File

def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        self.model_type = "fast"  # Default model type
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
        # self.pipe.enable_model_cpu_offload()

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        # Get configuration for current model
        config = MODEL_CONFIGS[self.model_type]
        
        # Parse resolution
        height, width = parse_resolution(input_data.resolution)
        
        # Set up generator for reproducibility
        seed = input_data.seed if input_data.seed != -1 else torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate image
        images = self.pipe(
            input_data.prompt,
            height=height,
            width=width,
            guidance_scale=config["guidance_scale"],
            num_inference_steps=config["num_inference_steps"],
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