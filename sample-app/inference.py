from typing import Optional
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import StableDiffusionXLPipeline
import torch

class AppInput(BaseAppInput):
    prompt: str = Field(
        ...,
        description="The text prompt to generate the image from",
        examples=["A majestic lion jumping from a big stone at night"]
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Negative prompt to avoid certain elements in the generated image",
        examples=["blurry, low quality, distorted"]
    )
    num_inference_steps: int = Field(
        50,
        description="Number of denoising steps",
        ge=1,
        le=100
    )
    guidance_scale: float = Field(
        7.5,
        description="Classifier-free guidance scale",
        ge=1.0,
        le=20.0
    )
    width: int = Field(
        1024,
        description="Width of the generated image",
        ge=256,
        le=2048
    )
    height: int = Field(
        1024,
        description="Height of the generated image",
        ge=256,
        le=2048
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    pipeline: Optional[StableDiffusionXLPipeline] = None

    async def setup(self):
        """Initialize the Stable Diffusion XL model."""
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipeline.to("cuda")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate an image based on the input prompt."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Generate the image
        image = self.pipeline(
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            width=input_data.width,
            height=input_data.height
        ).images[0]

        # Save the image
        output_path = "/tmp/generated_image.png"
        image.save(output_path)

        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        if self.pipeline:
            self.pipeline = None