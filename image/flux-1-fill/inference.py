from typing import Optional
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import torch
from diffusers import FluxFillPipeline
from PIL import Image

class AppInput(BaseAppInput):
    prompt: str = Field(..., description="Text prompt describing what to fill the masked area with")
    image: File = Field(..., description="Input image file to be filled")
    mask: File = Field(..., description="Mask image file indicating the area to be filled")
    guidance_scale: float = Field(30.0, description="Guidance scale for the generation process")
    num_inference_steps: int = Field(50, description="Number of denoising steps")
    max_sequence_length: int = Field(512, description="Maximum sequence length for the text encoder")
    seed: Optional[int] = Field(None, description="Random seed for reproducible results")

class AppOutput(BaseAppOutput):
    result: File = Field(..., description="Generated image with filled area")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the FluxFillPipeline model."""
        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Load input image and mask
        input_image = Image.open(input_data.image.path).convert("RGB")
        mask_image = Image.open(input_data.mask.path).convert("RGB")
        
        # Verify image and mask dimensions match
        if input_image.size != mask_image.size:
            raise ValueError(f"Image and mask dimensions do not match. Image: {input_image.size}, Mask: {mask_image.size}")
        
        # Generate the filled image
        result = self.pipe(
            prompt=input_data.prompt,
            image=input_image,
            mask_image=mask_image,
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
            max_sequence_length=input_data.max_sequence_length,
            generator=torch.Generator("cuda").manual_seed(input_data.seed if input_data.seed is not None else 0)
        ).images[0]
        
        # Save the result
        output_path = "/tmp/result.png"
        result.save(output_path)
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()