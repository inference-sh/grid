from typing import Optional
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

class AppInput(BaseAppInput):
    prompt: str = Field(..., description="Text prompt describing what to fill the masked area with")
    image: File = Field(..., description="Input image file to be filled")
    mask: File = Field(..., description="Mask image file indicating the area to be filled")
    guidance_scale: float = Field(7.5, description="Guidance scale for the generation process")
    num_inference_steps: int = Field(50, description="Number of denoising steps")
    strength: float = Field(0.99, description="Strength of the inpainting process (0.0 to 1.0)")
    model_url: str = Field("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", description="HuggingFace model URL for the inpainting model")
    seed: Optional[int] = Field(None, description="Random seed for reproducible results")

class AppOutput(BaseAppOutput):
    result: File = Field(..., description="Generated image with filled area")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the AutoPipelineForInpainting model with default URL."""
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        self.current_model_url = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Reinitialize the pipeline if a different model URL is provided
        if input_data.model_url != self.current_model_url:
            tmp = AutoPipelineForInpainting.from_pretrained(
                input_data.model_url,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("cuda")
            if self.pipe is not None and tmp is not None:
                del self.pipe
                torch.cuda.empty_cache()
            self.pipe = tmp
            self.current_model_url = input_data.model_url

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
            strength=input_data.strength,
            generator=torch.Generator("cuda").manual_seed(input_data.seed if input_data.seed is not None else 0)
        ).images[0]
        
        # Save the result
        output_path = "/tmp/result.png"
        result.save(output_path)
        
        return AppOutput(result=File(path=output_path))

