"""
Gemini Nano Banana 2.1 via Vertex AI.

Google's newest image model: better visual design, mask-based editing and subject
consistency than earlier Nano Banana models, and faster than Nano Banana 2.
"""

import mimetypes
from enum import Enum
from google.genai import types
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, TextMeta
from pydantic import Field
from typing import Optional, List

from .vertex_helper import (
    create_vertex_client,
    get_mime_type,
    OutputFormatEnum,
    SafetyToleranceEnum,
    calculate_dimensions,
    load_image_as_part,
    build_image_generation_config,
    setup_logger,
    resolve_aspect_ratio,
    retry_on_resource_exhausted,
    RetryConfig,
    process_image_response,
    raise_no_images_error,
    build_image_output_meta,
)


class AspectRatioEnum(str, Enum):
    """Aspect ratios supported by Nano Banana 2.1."""
    auto = "auto"
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_8_1 = "8:1"
    ratio_4_1 = "4:1"
    ratio_3_2 = "3:2"
    ratio_4_3 = "4:3"
    ratio_5_4 = "5:4"
    ratio_1_1 = "1:1"
    ratio_4_5 = "4:5"
    ratio_3_4 = "3:4"
    ratio_2_3 = "2:3"
    ratio_1_4 = "1:4"
    ratio_1_8 = "1:8"
    ratio_9_16 = "9:16"


class ResolutionEnum(str, Enum):
    """Resolutions supported by Nano Banana 2.1 (no 512)."""
    res_1k = "1K"
    res_2k = "2K"
    res_4k = "4K"



class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The prompt for image generation or editing. Describe what you want to create or change."
    )
    images: Optional[List[File]] = Field(
        None,
        description="Optional input images for editing (up to 14). When provided, the model edits them based on the prompt. Supported formats: JPEG, PNG, WebP, HEIC, HEIF"
    )
    context_files: Optional[List[File]] = Field(
        None,
        description="Optional video or PDF files passed as extra context (up to 10 videos). Supported formats: MP4, MOV, WebM, MPEG, PDF"
    )
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate.")
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_1_1,
        description="Aspect ratio. Supports extreme ratios like 1:4, 4:1, 1:8, 8:1. Use 'auto' to match the first input image."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.res_1k,
        description="Output resolution: 1K, 2K, 4K."
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output format for the generated images."
    )
    enable_google_search: bool = Field(
        default=False,
        description="Enable Google Search grounding (web and image search) for real-world subjects."
    )
    safety_tolerance: SafetyToleranceEnum = Field(
        default=SafetyToleranceEnum.block_none,
        description="Safety filter threshold."
    )
    retry_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Number of automatic retries on 429 rate limit errors."
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The generated or edited images")
    description: str = Field(default="", description="Text description from the model")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.model_id = "gemini-nano-banana-2.1"
        self.client = create_vertex_client()
        self.logger.info("Gemini Nano Banana 2.1 (Vertex AI) initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if input_data.images is not None:
                input_data.images = [img for img in input_data.images if img is not None]
            is_editing = input_data.images is not None and len(input_data.images) > 0

            if is_editing:
                if len(input_data.images) > 14:
                    raise RuntimeError("Supports up to 14 input images")
                for i, image in enumerate(input_data.images):
                    if not image.exists():
                        raise RuntimeError(f"Input image {i+1} does not exist: {image.path}")
                self.logger.info(f"Starting image editing: {input_data.prompt[:100]}...")
            else:
                self.logger.info(f"Starting image generation: {input_data.prompt[:100]}...")

            aspect_ratio_value = resolve_aspect_ratio(
                input_data.aspect_ratio.value,
                input_data.images if is_editing else None,
                self.logger
            )

            self.logger.info(f"Resolution: {input_data.resolution.value}, Aspect: {aspect_ratio_value}")

            contents = [input_data.prompt]
            if is_editing:
                for image in input_data.images:
                    contents.append(load_image_as_part(image.path, logger=self.logger))
            for f in input_data.context_files or []:
                if f is None:
                    continue
                if not f.exists():
                    raise RuntimeError(f"Context file does not exist: {f.path}")
                mime_type = get_mime_type(f.path, default=mimetypes.guess_type(f.path)[0] or f.content_type or "")
                if not (mime_type.startswith("video/") or mime_type == "application/pdf"):
                    raise RuntimeError(f"Unsupported context file type {mime_type}: use video or PDF")
                self.logger.info(f"Adding context file ({mime_type})")
                with open(f.path, "rb") as fh:
                    contents.append(types.Part.from_bytes(data=fh.read(), mime_type=mime_type))

            # Nano Banana 2.1 returns an API error if temperature, top_p or top_k is set.
            config = build_image_generation_config(
                aspect_ratio=aspect_ratio_value,
                resolution=input_data.resolution.value,
                output_format=input_data.output_format.value,
                temperature=None,
                top_p=None,
                top_k=None,
                safety_tolerance=input_data.safety_tolerance.value,
                enable_google_search=input_data.enable_google_search,
            )

            results = []
            retry_config = RetryConfig(max_attempts=input_data.retry_count + 1)

            for i in range(input_data.num_images):
                self.logger.info(f"Generating image {i+1}/{input_data.num_images}...")

                async def _generate():
                    return self.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    )

                response = await retry_on_resource_exhausted(_generate, config=retry_config, logger=self.logger)
                results.append(process_image_response(response, input_data.output_format.value, self.logger))

            output_images = [File(path=p) for r in results for p in r.image_paths]
            descriptions = [d for r in results for d in r.descriptions if d]

            if not output_images:
                raise_no_images_error(results)

            self.logger.info(f"Generated {len(output_images)} image(s)")

            width, height = calculate_dimensions(aspect_ratio_value, input_data.resolution.value)
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
                    extra={"web_search": input_data.enable_google_search},
                )
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")
