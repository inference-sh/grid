from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import torch
from diffusers import CogView4Pipeline
from PIL import Image
import os
import io
from typing import List
from pydantic import Field

class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text description of the image to generate. For best results, use detailed descriptions."
    )
    guidance_scale: float = Field(
        default=3.5,
        description="Controls how closely the image follows the prompt. Higher values adhere more strictly to the prompt.",
        ge=0.0,
        le=20.0
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps. More steps generally result in higher quality images but take longer.",
        ge=20,
        le=100
    )
    width: int = Field(
        default=1024,
        description="Width of the generated image in pixels. Must be a multiple of 32.",
        ge=512,
        le=2048
    )
    height: int = Field(
        default=1024,
        description="Height of the generated image in pixels. Must be a multiple of 32.",
        ge=512,
        le=2048
    )
    negative_prompt: str = Field(
        default="",
        description="Text description of elements to avoid in the generated image."
    )
    
    seed: int = Field(
        default=None,
        description="Random seed for reproducible image generation. Leave empty for random results."
    )

class AppOutput(BaseAppOutput):
    image: File = Field(
        description="The generated image in PNG format."
    )

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the CogView4 model and resources."""
        # Load the model with bfloat16 precision
        self.pipe = CogView4Pipeline.from_pretrained(
            "THUDM/CogView4-6B", 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        # self.pipe.transformer.to(memory_format=torch.channels_last)
        # self.pipe.transformer = torch.compile(self.pipe.transformer, mode="reduce-overhead", fullgraph=False)
        # Enable optimizations to reduce memory usage
        #self.pipe.enable_model_cpu_offload()
        #self.pipe.vae.enable_slicing()
        #self.pipe.vae.enable_tiling()
        
        # Optional: Load prompt optimization if needed
        self.optimize_prompt_enabled = False
        # You could add prompt optimization setup here if needed

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image from the input prompt using CogView4."""
        prompt = input_data.prompt
        
        # Set random seed if provided
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator("cuda").manual_seed(input_data.seed)
        
        # Generate the image
        output = self.pipe(
            prompt=prompt,
            negative_prompt=input_data.negative_prompt,
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
            width=input_data.width,
            height=input_data.height,
            num_images_per_prompt=1,
            generator=generator,
        )
        

        # Save the image to a temporary file
        output_path = "/tmp/generated_image.png"
        output.images[0].save(output_path)
        
        return AppOutput(image=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Free up GPU memory
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()