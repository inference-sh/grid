from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional
import torch
from diffusers import Flux2KleinPipeline
import os
import logging
from accelerate import Accelerator
from enum import Enum
from diffusers.utils import load_image
import random

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

class ModelType(str, Enum):
    FLUX_2_KLEIN_4B = "FLUX.2-klein-4B"
    FLUX_2_KLEIN_8B = "FLUX.2-klein-9B"

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    reference_images: Optional[list[File]] = Field(default=None, description="List of reference images to use for generation.")
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")

class AppOutput(BaseAppOutput):
    image: File = Field(description="The generated image.")

class AppSetup(BaseAppInput):
     model_id: ModelType = Field(default=ModelType.FLUX_2_KLEIN_4B, description="Model to load") 

class App(BaseApp):
    async def setup(self, config: AppSetup):
        logging.basicConfig(level=logging.INFO)
        self.original_model_id = config.model_id
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/" + self.original_model_id,
            torch_dtype=torch.bfloat16,
        )
        # self.pipeline.enable_model_cpu_offload()
        self.pipeline = self.pipeline.to(device)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        
        reference_images = [load_image(image.path) for image in input_data.reference_images] if input_data.reference_images else None

        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=device).manual_seed(input_data.seed)

        image = self.pipeline(
            prompt=input_data.prompt,
            guidance_scale=1.0,
            num_inference_steps=4,
            height=input_data.height,
            width=input_data.width,
            image=reference_images,
            generator=generator,
        ).images[0]
        
        random_int = random.randint(0, 1000000)
        output_path = f"/tmp/generated_image_{random_int}.png"
        image.save(output_path)
        return AppOutput(image=File(path=output_path))