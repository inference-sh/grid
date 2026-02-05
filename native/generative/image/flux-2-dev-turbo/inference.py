from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional
import torch
from diffusers import Flux2Pipeline
import os
import logging
from accelerate import Accelerator
from enum import Enum
from diffusers.utils import load_image

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    reference_images: Optional[list[File]] = Field(default=None, description="List of reference images to use for generation.")
    guidance_scale: float = Field(default=3.5, description="The guidance scale.")
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image.")

class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.original_model_id = "black-forest-labs/FLUX.2-dev"
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.original_model_id,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.load_lora_weights(
            "fal/FLUX.2-dev-Turbo", 
            weight_name="flux.2-turbo-lora.safetensors"
        )
        # self.pipeline = self.pipeline.to(device)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        
        reference_images = [load_image(image.path) for image in input_data.reference_images] if input_data.reference_images else None

        image = self.pipeline(
            prompt=input_data.prompt,
            sigmas=TURBO_SIGMAS,
            guidance_scale=input_data.guidance_scale,
            height=input_data.height,
            width=input_data.width,
            image=reference_images,
        ).images[0]
        
        output_path = "/tmp/generated_image.png"
        image.save(output_path)
        return AppOutput(image_output=File(path=output_path))