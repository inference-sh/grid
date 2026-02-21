from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, TextMeta
from pydantic import Field
from typing import Optional, List

from .vertex_helper import (
    create_vertex_client,
    OutputFormatExtendedEnum,
    AspectRatioAutoEnum,
    SafetyToleranceEnum,
    ResolutionEnum,
    calculate_dimensions,
    load_image_as_part,
    build_image_generation_config,
    setup_logger,
    resolve_aspect_ratio,
    retry_on_resource_exhausted,
    process_image_response,
    raise_no_images_error,
    build_image_output_meta,
)


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
    aspect_ratio: AspectRatioAutoEnum = Field(
        default=AspectRatioAutoEnum.ratio_1_1,
        description="Aspect ratio for the output image. Use 'auto' to automatically match the first input image's aspect ratio. Default: 1:1"
    )
    # resolution: ResolutionEnum = Field(
    #     default=ResolutionEnum.res_1k,
    #     description="Output resolution. Options: 1K, 2K, 4K. Default: 1K"
    # )
    output_format: OutputFormatExtendedEnum = Field(
        default=OutputFormatExtendedEnum.png,
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
    async def setup(self):
        """Initialize model and configuration."""
        self.logger = setup_logger(__name__)
        self.model_id = "gemini-2.5-flash-image"
        self.client = create_vertex_client()
        self.logger.info("Gemini 2.5 Flash Image (Vertex AI) initialized successfully")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit images using Gemini 2.5 Flash model via Vertex AI."""
        try:
            is_editing = input_data.images is not None and len(input_data.images) > 0

            if is_editing:
                if len(input_data.images) > 14:
                    raise RuntimeError("Gemini 2.5 Flash supports up to 14 input images")

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

            resolution = ResolutionEnum.res_1k
            self.logger.info(f"Resolution: {resolution.value}, Aspect ratio: {aspect_ratio_value}")
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
                resolution=resolution.value,
                temperature=input_data.temperature,
                top_p=input_data.top_p,
                top_k=input_data.top_k,
                max_output_tokens=input_data.max_output_tokens,
                safety_tolerance=input_data.safety_tolerance.value,
                enable_google_search=input_data.enable_google_search,
            )

            # Generate images (one API call per image)
            results = []

            for i in range(input_data.num_images):
                self.logger.info(f"Generating image {i+1}/{input_data.num_images}...")

                async def _generate():
                    return self.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    )

                response = await retry_on_resource_exhausted(_generate, logger=self.logger)
                result = process_image_response(response, input_data.output_format.value, self.logger)
                results.append(result)

            # Collect all images and descriptions
            output_images = [File(path=p) for r in results for p in r.image_paths]
            descriptions = [d for r in results for d in r.descriptions]

            if not output_images:
                raise_no_images_error(results)

            self.logger.info(f"Successfully generated {len(output_images)} image(s)")

            width, height = calculate_dimensions(aspect_ratio_value, resolution.value)
            meta = build_image_output_meta(results, width, height)

            return AppOutput(
                images=output_images,
                description="\n".join(descriptions) if descriptions else "",
                output_meta=OutputMeta(
                    inputs=[TextMeta(**m) for m in meta["inputs"]],
                    outputs=[
                        TextMeta(**m) if m["type"] == "text" else ImageMeta(**{k: v for k, v in m.items() if k != "type"})
                        for m in meta["outputs"]
                    ],
                )
            )

        except Exception as e:
            self.logger.error(f"Error during image generation/editing: {e}")
            raise RuntimeError(f"Image generation/editing failed: {str(e)}")
