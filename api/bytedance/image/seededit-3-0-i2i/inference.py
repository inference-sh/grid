"""
SeedEdit 3.0 I2I - BytePlus Image Editing

Edit and transform existing images using text prompts.
Uses direct HTTP requests (SDK doesn't support seededit yet).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import ClassVar, Optional
import logging
import os
import requests

from .byteplus_helper import download_image


class AppInput(BaseAppInput):
    """Input schema for SeedEdit 3.0 image editing."""

    prompt: str = Field(
        description="Text prompt describing the desired edit or transformation. Be specific about what changes you want.",
        examples=["Change the background to a sunset beach", "Make the person wear a red dress"]
    )
    image: File = Field(
        description="Source image to edit. Required for image-to-image editing."
    )
    guidance_scale: float = Field(
        default=5.5,
        ge=1.0,
        le=20.0,
        description="Guidance scale for edit strength (1.0-20.0). Higher values follow the prompt more strictly."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results. If not provided, a random seed is used."
    )
    watermark: bool = Field(
        default=False,
        description="Whether to add a watermark to the generated image."
    )


class AppOutput(BaseAppOutput):
    """Output schema for SeedEdit 3.0 image editing."""

    image: File = Field(description="The edited image file.")


class App(BaseApp):
    """SeedEdit 3.0 image editing application using direct HTTP API."""

    API_URL: ClassVar[str] = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    MODEL_ID: ClassVar[str] = "seededit-3-0-i2i-250628"

    async def setup(self, metadata):
        """Initialize the API configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress noisy httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Get API key
        self.api_key = os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY environment variable is required")

        self.logger.info(f"SeedEdit 3.0 I2I initialized with model: {self.MODEL_ID}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Edit image using SeedEdit 3.0 via direct HTTP."""
        try:
            self.logger.info("Starting image editing")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Guidance scale: {input_data.guidance_scale}, Watermark: {input_data.watermark}")

            # Validate input image
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

            # Build request payload matching curl example exactly
            payload = {
                "model": self.MODEL_ID,
                "prompt": input_data.prompt,
                "image": input_data.image.uri,
                "response_format": "url",
                "size": "adaptive",
                "guidance_scale": input_data.guidance_scale,
                "watermark": input_data.watermark,
            }

            # Add seed if provided
            if input_data.seed is not None:
                payload["seed"] = input_data.seed

            # Make direct HTTP request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            self.logger.info(f"Calling API: {self.API_URL}")
            response = requests.post(self.API_URL, json=payload, headers=headers, timeout=120)

            # Check for errors
            if response.status_code != 200:
                error_detail = response.text
                self.logger.error(f"API error {response.status_code}: {error_detail}")
                raise RuntimeError(f"API error {response.status_code}: {error_detail}")

            result = response.json()

            # Extract image URL from response
            if "data" not in result or len(result["data"]) == 0:
                raise RuntimeError(f"No image data in response: {result}")

            image_url = result["data"][0].get("url")
            if not image_url:
                self.logger.error(f"Could not extract image URL from result: {result}")
                raise RuntimeError("Failed to get image URL from response")

            # Download image
            image_path = download_image(image_url, self.logger)

            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=0,  # Adaptive - unknown until downloaded
                        height=0,
                        extra={
                            "mode": "image-editing",
                            "guidance_scale": input_data.guidance_scale,
                            "seed": input_data.seed,
                            "watermark": input_data.watermark,
                        }
                    )
                ]
            )

            self.logger.info(f"Image edited successfully: {image_path}")

            return AppOutput(
                image=File(path=image_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during image editing: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}")
