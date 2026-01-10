from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import tempfile
import os
import logging
import math

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.oauth2.credentials import Credentials


class OutputFormatEnum(str, Enum):
    """Output format options."""
    png = "png"
    jpeg = "jpeg"
    webp = "webp"
    heic = "heic"
    heif = "heif"


class AspectRatioEnum(str, Enum):
    """Aspect ratio options."""
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_3_2 = "3:2"
    ratio_4_3 = "4:3"
    ratio_5_4 = "5:4"
    ratio_1_1 = "1:1"
    ratio_4_5 = "4:5"
    ratio_3_4 = "3:4"
    ratio_2_3 = "2:3"
    ratio_9_16 = "9:16"



class SafetyToleranceEnum(str, Enum):
    """Safety filter thresholds."""
    block_none = "BLOCK_NONE"
    block_low_and_above = "BLOCK_LOW_AND_ABOVE"
    block_medium_and_above = "BLOCK_MEDIUM_AND_ABOVE"
    block_only_high = "BLOCK_ONLY_HIGH"
    off = "OFF"


class ResolutionEnum(str, Enum):
    """Resolution options."""
    res_1k = "1K"
    res_2k = "2K"
    res_4k = "4K"


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The prompt for image generation or editing. Describe what you want to create or change."
    )
    images: Optional[List[File]] = Field(
        None,
        description="Optional list of input images for editing (up to 14 images). Max file size: 7MB (inline). Supported formats: PNG, JPEG, WEBP, HEIC, HEIF."
    )
    num_images: int = Field(
        1,
        description="Number of images to generate.",
        ge=1,
        le=4
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_1_1,
        description="Aspect ratio for the output image. Default: 1:1"
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.res_1k,
        description="Output resolution. Options: 1K, 2K, 4K. Default: 1K"
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output format for the generated images."
    )
    enable_google_search: bool = Field(
        default=False,
        description="Enable Google Search grounding for real-time information (weather, news, etc.)"
    )
    temperature: float = Field(
        default=1.0,
        description="Controls randomness in token selection. range: 0.0 - 2.0. Default: 1.0",
        ge=0.0,
        le=2.0
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling probability. Range: 0.0 - 1.0. Default: 0.95",
        ge=0.0,
        le=1.0
    )
    top_k: int = Field(
        default=64,
        description="Top-k sampling. Fixed at 64 for this model.",
    )
    max_output_tokens: int = Field(
        default=32768,
        description="Maximum number of tokens to generate. Max: 32768"
    )
    safety_tolerance: SafetyToleranceEnum = Field(
        default=SafetyToleranceEnum.block_medium_and_above,
        description="Safety filter threshold for all categories."
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
        self.model_id = "gemini-3-pro-image-preview"

        # Get Vertex AI credentials from environment
        access_token = os.environ.get("GCP_ACCESS_TOKEN")
        project = os.environ.get("GCP_PROJECT_NUMBER")
        
        if not access_token:
            raise RuntimeError(
                "GCP_ACCESS_TOKEN environment variable is required for Vertex AI access."
            )
        if not project:
            raise RuntimeError(
                "GCP_PROJECT_NUMBER environment variable is required for Vertex AI access."
            )

        # Create credentials from access token
        credentials = Credentials(token=access_token)
        
        # Create Vertex AI client
        self.client = genai.Client(
            vertexai=True,
            project=project,
            # location="global", # gemini-3-pro-image-preview model only supports global location?
            credentials=credentials,
            http_options=HttpOptions(api_version="v1")
        )

        self.logger.info("Gemini 3 Pro Image Preview (Vertex AI) initialized successfully")

    def _get_dimensions(self, aspect_ratio: str, resolution: str) -> tuple[int, int]:
        """Estimate dimensions based on aspect ratio and resolution string."""
        # Baseline 1K = 1024x1024 approx
        base = 1024
        if resolution == "2K":
            base = 2048
        elif resolution == "4K":
            base = 4096
            
        # Aspect ratio multipliers (approximate)
        # We assume base is the side length of a square, and we scale width/height to preserve area?
        # Or more likely with GenAI, '1024' is the standard side for 1:1.
        # For non-square, usually one side increases and other decreases or stays same.
        # We will use equal-area approximation: w * h = base * base.
        # w / h = ratio.
        # h * ratio * h = base^2 => h^2 = base^2 / ratio => h = base / sqrt(ratio)
        
        ar_map = {
            "1:1": 1.0,
            "16:9": 16/9,
            "21:9": 21/9,
            "3:2": 3/2,
            "4:3": 4/3,
            "5:4": 5/4,
            "4:5": 4/5,
            "3:4": 3/4,
            "2:3": 2/3,
            "9:16": 9/16,
        }
        
        ratio_val = ar_map.get(aspect_ratio, 1.0)
        
        height = int(base / math.sqrt(ratio_val))
        width = int(height * ratio_val)
        
        return width, height

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.heic': 'image/heic',
            '.heif': 'image/heif',
        }
        return mime_types.get(ext, 'image/png')

    def _load_image_as_part(self, file_path: str) -> types.Part:
        """Load an image file and return it as a Gemini Part."""
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        mime_type = self._get_mime_type(file_path)
        return types.Part.from_bytes(data=image_data, mime_type=mime_type)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Gemini 3 Pro Image Preview model via Vertex AI."""
        try:
            # Determine if this is generation or editing
            is_editing = input_data.images is not None and len(input_data.images) > 0

            if is_editing:
                # Validate input files for editing
                if len(input_data.images) > 14:
                    raise RuntimeError("Gemini 3 Pro Image Preview supports up to 14 input images")
                
                for i, image in enumerate(input_data.images):
                    if not image.exists():
                        raise RuntimeError(f"Input image {i+1} does not exist at path: {image.path}")
                
                self.logger.info(f"Starting image editing with prompt: {input_data.prompt[:100]}...")
                self.logger.info(f"Processing {len(input_data.images)} input image(s)")
            else:
                self.logger.info(f"Starting image generation with prompt: {input_data.prompt[:100]}...")

            self.logger.info(f"Resolution: {input_data.resolution.value}, Aspect ratio: {input_data.aspect_ratio.value}")
            self.logger.info(f"Requesting {input_data.num_images} output image(s)")

            # Build content parts
            contents = [input_data.prompt]
            
            if is_editing:
                # Add images for editing
                for image in input_data.images:
                    image_part = self._load_image_as_part(image.path)
                    contents.append(image_part)

            # Configure generation settings
            image_config = types.ImageConfig(
                aspect_ratio=input_data.aspect_ratio.value,
                image_size=input_data.resolution.value,
            )

            config_kwargs = {
                'response_modalities': ['TEXT', 'IMAGE'],
                'image_config': image_config,
                'temperature': input_data.temperature,
                'top_p': input_data.top_p,
                'top_k': input_data.top_k,
                'max_output_tokens': input_data.max_output_tokens,
                'safety_settings': [
                    types.SafetySetting(
                        category=category,
                        threshold=input_data.safety_tolerance.value
                    ) for category in [
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT"
                    ]
                ]
            }

            # Add Google Search tool if enabled
            if input_data.enable_google_search:
                config_kwargs['tools'] = [{"google_search": {}}]
                self.logger.info("Google Search grounding enabled")

            config = types.GenerateContentConfig(**config_kwargs)

            # Generate images (one API call per image)
            output_images = []
            descriptions = []

            for i in range(input_data.num_images):
                self.logger.info(f"Generating image {i+1}/{input_data.num_images}...")

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )

                # Process response parts
                for part in response.candidates[0].content.parts:
                    # Skip thought parts (used internally by the model)
                    if hasattr(part, 'thought') and part.thought:
                        continue
                        
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

            width, height = self._get_dimensions(input_data.aspect_ratio.value, input_data.resolution.value)
            
            output_meta_images = []
            for _ in output_images:
                output_meta_images.append(ImageMeta(
                    width=width,
                    height=height
                ))

            return AppOutput(
                images=output_images,
                description="\n".join(descriptions) if descriptions else "",
                output_meta=OutputMeta(outputs=output_meta_images)
            )

        except Exception as e:
            self.logger.error(f"Error during image generation/editing: {e}")
            raise RuntimeError(f"Image generation/editing failed: {str(e)}")
