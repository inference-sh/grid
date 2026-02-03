from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List

from .vertex_helper import (
    create_vertex_client,
    OutputFormatEnum,
    AspectRatioAutoEnum,
    ResolutionEnum,
    calculate_dimensions,
    load_image_as_part,
    save_image_to_temp,
    build_image_generation_config,
    setup_logger,
    resolve_aspect_ratio,
    retry_on_resource_exhausted,
)


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The prompt for image generation or editing. Describe what you want to create or change."
    )
    images: Optional[List[File]] = Field(
        None,
        description="Optional list of input images for editing (up to 14 images). When provided, the model will edit these images based on the prompt. When not provided, the model will generate new images from the text prompt. Supported formats: JPEG, PNG, WebP"
    )
    num_images: int = Field(
        1,
        description="Number of images to generate.",
        ge=1,
        le=4
    )
    aspect_ratio: AspectRatioAutoEnum = Field(
        default=AspectRatioAutoEnum.ratio_1_1,
        description="Aspect ratio for the output image. Use 'auto' to automatically match the first input image's aspect ratio. Default: 1:1"
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


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The generated or edited images")
    description: str = Field(default="", description="Text description or response from the model")


class App(BaseApp):
    async def setup(self):
        """Initialize model and configuration."""
        self.logger = setup_logger(__name__)
        self.model_id = "gemini-3-pro-image-preview"
        self.client = create_vertex_client()
        self.logger.info("Gemini 3 Pro Image Preview (Vertex AI) initialized successfully")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit images using Gemini 3 Pro Image Preview model via Vertex AI."""
        try:
            is_editing = input_data.images is not None and len(input_data.images) > 0

            if is_editing:
                if len(input_data.images) > 14:
                    raise RuntimeError("Gemini 3 Pro Image Preview supports up to 14 input images")

                for i, image in enumerate(input_data.images):
                    if not image.exists():
                        raise RuntimeError(f"Input image {i+1} does not exist at path: {image.path}")

                self.logger.info(f"Starting image editing with prompt: {input_data.prompt[:100]}...")
                self.logger.info(f"Processing {len(input_data.images)} input image(s)")
            else:
                self.logger.info(f"Starting image generation with prompt: {input_data.prompt[:100]}...")

            # Resolve aspect ratio (handle "auto")
            aspect_ratio_value = resolve_aspect_ratio(
                input_data.aspect_ratio.value,
                input_data.images if is_editing else None,
                self.logger
            )

            self.logger.info(f"Resolution: {input_data.resolution.value}, Aspect ratio: {aspect_ratio_value}")
            self.logger.info(f"Requesting {input_data.num_images} output image(s)")

            # Build content parts
            contents = [input_data.prompt]

            if is_editing:
                for image in input_data.images:
                    image_part = load_image_as_part(image.path, logger=self.logger)
                    contents.append(image_part)

            # Configure generation settings
            config = build_image_generation_config(
                aspect_ratio=aspect_ratio_value,
                resolution=input_data.resolution.value,
                enable_google_search=input_data.enable_google_search,
            )

            # Generate images (one API call per image)
            output_images = []
            descriptions = []

            for i in range(input_data.num_images):
                self.logger.info(f"Generating image {i+1}/{input_data.num_images}...")

                async def _generate():
                    return self.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    )

                response = await retry_on_resource_exhausted(_generate, logger=self.logger)

                # Process response parts
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        continue

                    if part.text is not None:
                        descriptions.append(part.text)
                        self.logger.info(f"Model response: {part.text[:200]}...")
                    elif part.inline_data is not None:
                        image_path = save_image_to_temp(
                            part.inline_data.data,
                            input_data.output_format.value
                        )
                        output_images.append(File(path=image_path))
                        self.logger.info(f"Saved image to {image_path}")

            if not output_images:
                raise RuntimeError("No images were generated")

            self.logger.info(f"Successfully generated {len(output_images)} image(s)")

            width, height = calculate_dimensions(aspect_ratio_value, input_data.resolution.value)

            output_meta_images = [
                ImageMeta(width=width, height=height)
                for _ in output_images
            ]

            return AppOutput(
                images=output_images,
                description="\n".join(descriptions) if descriptions else "",
                output_meta=OutputMeta(
                    outputs=output_meta_images,
                    extra={
                        "web_search": input_data.enable_google_search,
                    }
                )
            )

        except Exception as e:
            self.logger.error(f"Error during image generation/editing: {e}")
            raise RuntimeError(f"Image generation/editing failed: {str(e)}")
