from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from typing import Optional
from .omni_zero import OmniZeroSingle
from PIL import Image

class AppInput(BaseAppInput):
    seed: int = 42
    prompt: str
    negative_prompt: str = "blurry, out of focus"
    guidance_scale: float = 3.0
    number_of_steps: int = 10
    base_image: File
    base_image_strength: float = 0.15
    composition_image: File
    composition_image_strength: float = 1.0
    style_image: File
    style_image_strength: float = 1.0
    identity_image: File
    identity_image_strength: float = 1.0
    depth_image: Optional[File] = None
    depth_image_strength: float = 0.5

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.omni_zero = OmniZeroSingle(
            base_model="frankjoshua/albedobaseXL_v13",
        )

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Load and process images
        base_image = Image.open(input_data.base_image.path)
        composition_image = Image.open(input_data.composition_image.path)
        style_image = Image.open(input_data.style_image.path)
        identity_image = Image.open(input_data.identity_image.path)
        depth_image = Image.open(input_data.depth_image.path) if input_data.depth_image else None

        # Generate image
        images = self.omni_zero.generate(
            seed=input_data.seed,
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            guidance_scale=input_data.guidance_scale,
            number_of_images=1,  # We only generate one image
            number_of_steps=input_data.number_of_steps,
            base_image=base_image,
            base_image_strength=input_data.base_image_strength,
            composition_image=composition_image,
            composition_image_strength=input_data.composition_image_strength,
            style_image=style_image,
            style_image_strength=input_data.style_image_strength,
            identity_image=identity_image,
            identity_image_strength=input_data.identity_image_strength,
            depth_image=depth_image,
            depth_image_strength=input_data.depth_image_strength,
        )

        # Save the generated image
        output_path = "/tmp/" + input_data.identity_image.filename + ".png"
        images[0].save(output_path)
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        pass