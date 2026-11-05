"""GPT Image 2.6 Flare: image generation and editing via OpenAI's Images API."""

from inferencesh import BaseApp

from .gpt_image_26 import ImageInput, ImageOutput, run_image
from .openai_helper import create_openai_client, setup_logger

MODEL = "gpt-image-2.6-flare"


class AppInput(ImageInput):
    """Input schema for GPT Image 2.6 Flare."""


class AppOutput(ImageOutput):
    """Output schema for GPT Image 2.6 Flare."""


class App(BaseApp):
    """GPT Image 2.6 Flare image generation application."""

    async def setup(self):
        self.logger = setup_logger(__name__)
        self.client = create_openai_client()

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit images."""
        images, output_meta = await run_image(self.client, MODEL, input_data, self.logger)
        return AppOutput(images=images, output_meta=output_meta)
