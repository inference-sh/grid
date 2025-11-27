from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import fal_client
import tempfile
import os
import logging
import requests


class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt to generate an image from.")
    height: int = Field(default=1024, description="The height in pixels of the generated image.")
    width: int = Field(default=1024, description="The width in pixels of the generated image.")
    num_inference_steps: int = Field(default=30, description="The number of inference steps.")
    guidance_scale: float = Field(default=3.5, description="The guidance scale.")
    reference_images: Optional[list[File]] = Field(default=None, description="List of reference images to use for generation (enables edit mode).")
    seed: Optional[int] = Field(default=None, description="The seed for random generation.")


class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated image.")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Model endpoints
        self.text_to_image_model = "fal-ai/flux-2"
        self.edit_model = "fal-ai/flux-2/edit"

        self.logger.info("Flux 2 model initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate image using Flux 2 model via fal."""
        try:
            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError("FAL_KEY environment variable is required for model access.")

            fal_client.api_key = api_key

            # Determine which model to use based on reference images
            if input_data.reference_images:
                model_id = self.edit_model
                image_urls = [img.uri for img in input_data.reference_images]
                self.logger.info(f"Using edit mode with {len(image_urls)} reference image(s)")
            else:
                model_id = self.text_to_image_model
                image_urls = None
                self.logger.info("Using text-to-image mode")

            # Prepare request arguments
            arguments = {
                "prompt": input_data.prompt,
                "image_size": {
                    "width": input_data.width,
                    "height": input_data.height
                },
                "num_inference_steps": input_data.num_inference_steps,
                "guidance_scale": input_data.guidance_scale,
            }

            if input_data.seed is not None:
                arguments["seed"] = input_data.seed

            if image_urls:
                arguments["image_urls"] = image_urls

            self.logger.info(f"Generating image for prompt: '{input_data.prompt}'")

            # Define progress callback
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self.logger.info(f"Model: {log['message']}")

            # Run model inference
            result = fal_client.subscribe(
                model_id,
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            self.logger.info("Image generation completed successfully")

            # Process the generated image
            image_url = result["images"][0]["url"]
            self.logger.info("Processing generated image output...")

            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image_path = tmp_file.name

            # Download image content
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            with open(image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info("Image processing completed successfully")

            return AppOutput(image_output=File(path=image_path))

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        self.logger.info("Flux 2 model unloaded successfully")
