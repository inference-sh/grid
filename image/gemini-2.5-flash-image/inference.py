from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List
from enum import Enum
import tempfile
import os
import logging
import base64

from google import genai
from google.genai import types


class OutputFormatEnum(str, Enum):
    """Output format options."""
    jpeg = "jpeg"
    png = "png"


class AspectRatioEnum(str, Enum):
    """Aspect ratio options for generated images."""
    aspect_21_9 = "21:9"
    aspect_1_1 = "1:1"
    aspect_4_3 = "4:3"
    aspect_3_2 = "3:2"
    aspect_2_3 = "2:3"
    aspect_5_4 = "5:4"
    aspect_4_5 = "4:5"
    aspect_3_4 = "3:4"
    aspect_16_9 = "16:9"
    aspect_9_16 = "9:16"


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The prompt for image generation or editing. Describe what you want to create or change."
    )
    images: Optional[List[File]] = Field(
        None,
        description="Optional list of input images for editing. When provided, the model will edit these images based on the prompt. When not provided, the model will generate new images from the text prompt. Supported formats: JPEG, PNG, WebP"
    )
    num_images: int = Field(
        1,
        description="Number of images to generate.",
        ge=1,
        le=4
    )
    output_format: OutputFormatEnum = Field(
        OutputFormatEnum.png,
        description="Output format for the generated images."
    )
    aspect_ratio: Optional[AspectRatioEnum] = Field(
        None,
        description="Aspect ratio for generated images. Default is 1:1 for generation, or matches input image aspect ratio for editing."
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The generated or edited images")
    description: str = Field(default="", description="Text description or response from the model")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.metadata = metadata
        self.model_id = "gemini-2.5-flash-image"
        self.logger.info("Gemini 2.5 Flash Image initialized successfully")

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
        }
        return mime_types.get(ext, 'image/png')

    def _load_image_as_part(self, file_path: str) -> types.Part:
        """Load an image file and return it as a Gemini Part."""
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        mime_type = self._get_mime_type(file_path)
        return types.Part.from_bytes(data=image_data, mime_type=mime_type)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Gemini 2.5 Flash Image model."""
        try:
            # Set up API key from environment
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY environment variable is required for model access."
                )

            # Create Gemini client
            client = genai.Client(api_key=api_key)

            # Determine if this is generation or editing
            is_editing = input_data.images is not None and len(input_data.images) > 0

            if is_editing:
                # Validate input files for editing
                for i, image in enumerate(input_data.images):
                    if not image.exists():
                        raise RuntimeError(f"Input image {i+1} does not exist at path: {image.path}")
                
                self.logger.info(f"Starting image editing with prompt: {input_data.prompt[:100]}...")
                self.logger.info(f"Processing {len(input_data.images)} input image(s)")
            else:
                self.logger.info(f"Starting image generation with prompt: {input_data.prompt[:100]}...")

            self.logger.info(f"Requesting {input_data.num_images} output image(s)")

            # Build content parts
            contents = [input_data.prompt]
            
            if is_editing:
                # Add images for editing
                for image in input_data.images:
                    image_part = self._load_image_as_part(image.path)
                    contents.append(image_part)

            # Configure generation settings
            config_kwargs = {
                'response_modalities': ['TEXT', 'IMAGE'],
            }
            
            # Add image config if aspect ratio is specified
            if input_data.aspect_ratio is not None:
                config_kwargs['image_config'] = types.ImageConfig(
                    aspect_ratio=input_data.aspect_ratio.value,
                )

            config = types.GenerateContentConfig(**config_kwargs)

            # Generate images (may need multiple calls for multiple images)
            output_images = []
            descriptions = []

            for i in range(input_data.num_images):
                self.logger.info(f"Generating image {i+1}/{input_data.num_images}...")
                
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )

                # Process response parts
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        descriptions.append(part.text)
                        self.logger.info(f"Model response: {part.text[:200]}...")
                    elif part.inline_data is not None:
                        # Save image to temp file
                        file_extension = f".{input_data.output_format.value}"
                        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                            image_path = tmp_file.name
                        
                        # Decode and save image
                        image_bytes = part.inline_data.data
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        output_images.append(File(path=image_path))
                        self.logger.info(f"Saved image to {image_path}")

            if not output_images:
                raise RuntimeError("No images were generated")

            self.logger.info(f"Successfully generated {len(output_images)} image(s)")

            return AppOutput(
                images=output_images,
                description="\n".join(descriptions) if descriptions else ""
            )

        except Exception as e:
            self.logger.error(f"Error during image generation/editing: {e}")
            raise RuntimeError(f"Image generation/editing failed: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        self.logger.info("Gemini 2.5 Flash Image unloaded successfully")
