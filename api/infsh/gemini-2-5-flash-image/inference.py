"""
Gemini 2.5 Flash Image with fal.ai fallback.

Uses Vertex AI with automatic fallback to fal.ai (nano-banana) on rate limits.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, TextMeta
from pydantic import Field
from typing import Optional, List
import asyncio
import random

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
    is_resource_exhausted_error,
    process_image_response,
    raise_no_images_error,
    build_image_output_meta,
)
from .fal_helper import setup_fal_client, run_fal_model, download_file


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The prompt for image generation or editing. Describe what you want to create or change."
    )
    images: Optional[List[File]] = Field(
        None,
        description="Optional list of input images for editing (up to 14 images). Supported formats: PNG, JPEG, WEBP, HEIC, HEIF."
    )
    num_images: int = Field(
        1,
        description="Number of images to generate.",
        ge=1,
        le=4
    )
    aspect_ratio: AspectRatioAutoEnum = Field(
        default=AspectRatioAutoEnum.ratio_1_1,
        description="Aspect ratio for the output image. Use 'auto' to match input image. Default: 1:1"
    )
    output_format: OutputFormatExtendedEnum = Field(
        default=OutputFormatExtendedEnum.png,
        description="Output format for the generated images."
    )
    enable_google_search: bool = Field(
        default=False,
        description="Enable Google Search grounding for real-time information."
    )
    safety_tolerance: SafetyToleranceEnum = Field(
        default=SafetyToleranceEnum.block_none,
        description="Safety filter threshold."
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The generated or edited images")
    description: str = Field(default="", description="Text description or response from the model")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.model_id = "gemini-2.5-flash-image"
        self.fal_text_model = "fal-ai/nano-banana"
        self.fal_edit_model = "fal-ai/nano-banana/edit"
        self.client = create_vertex_client()
        self.logger.info("Gemini 2.5 Flash Image (with fal fallback) initialized")

    async def _generate_fal(self, input_data: AppInput, aspect_ratio_value: str) -> list:
        """Generate using fal.ai as fallback."""
        setup_fal_client()
        is_editing = input_data.images and len(input_data.images) > 0
        model_id = self.fal_edit_model if is_editing else self.fal_text_model

        from dataclasses import dataclass, field
        @dataclass
        class FalResult:
            image_paths: list = field(default_factory=list)
            descriptions: list = field(default_factory=list)

        results = []
        for i in range(input_data.num_images):
            self.logger.info(f"fal.ai: generating image {i+1}/{input_data.num_images}")

            # Map Vertex safety tolerance to fal (1=strict, 6=permissive)
            safety_map = {
                "BLOCK_NONE": "6",
                "BLOCK_ONLY_HIGH": "5",
                "BLOCK_MEDIUM_AND_ABOVE": "3",
                "BLOCK_LOW_AND_ABOVE": "2",
            }
            fal_safety = safety_map.get(input_data.safety_tolerance.value, "4")

            request = {
                "prompt": input_data.prompt,
                "num_images": 1,
                "output_format": input_data.output_format.value,
                "limit_generations": True,
                "safety_tolerance": fal_safety,
            }
            if aspect_ratio_value != "auto":
                request["aspect_ratio"] = aspect_ratio_value
            if is_editing:
                request["image_urls"] = [img.uri for img in input_data.images]

            result = run_fal_model(model_id, request, self.logger, with_logs=False)

            image_paths = []
            for img_data in result.get("images", []):
                path = download_file(img_data["url"], f".{input_data.output_format.value}", self.logger)
                image_paths.append(path)

            results.append(FalResult(
                image_paths=image_paths,
                descriptions=[result.get("description", "")]
            ))

        return results

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
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

            resolution = ResolutionEnum.res_1k
            self.logger.info(f"Resolution: {resolution.value}, Aspect: {aspect_ratio_value}")

            contents = [input_data.prompt]
            if is_editing:
                for image in input_data.images:
                    contents.append(load_image_as_part(image.path, logger=self.logger))

            config = build_image_generation_config(
                aspect_ratio=aspect_ratio_value,
                resolution=resolution.value,
                safety_tolerance=input_data.safety_tolerance.value,
                enable_google_search=input_data.enable_google_search,
            )

            results = []
            provider = "vertex"
            use_fal = False

            for i in range(input_data.num_images):
                if use_fal:
                    break

                self.logger.info(f"Generating image {i+1}/{input_data.num_images}...")

                # Try Vertex with 2 retries, then fallback to fal
                for attempt in range(1, 3):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_id,
                            contents=contents,
                            config=config,
                        )
                        result = process_image_response(response, input_data.output_format.value, self.logger)
                        results.append(result)
                        break
                    except Exception as e:
                        if is_resource_exhausted_error(e):
                            if attempt < 2:
                                delay = random.uniform(0.3, 0.6) * attempt
                                self.logger.warning(f"Vertex 429 attempt {attempt}/2, retry in {delay:.1f}s")
                                await asyncio.sleep(delay)
                            else:
                                self.logger.warning("Vertex 429 after 2 attempts, falling back to fal.ai")
                                use_fal = True
                                break
                        else:
                            raise

            if use_fal:
                results = await self._generate_fal(input_data, aspect_ratio_value)
                provider = "fal"

            output_images = [File(path=p) for r in results for p in r.image_paths]
            descriptions = [d for r in results for d in r.descriptions if d]

            if not output_images:
                raise_no_images_error(results)

            self.logger.info(f"Generated {len(output_images)} image(s) via {provider}")

            width, height = calculate_dimensions(aspect_ratio_value, resolution.value)
            meta = build_image_output_meta(results, width, height)

            return AppOutput(
                images=output_images,
                description="\n".join(descriptions) if descriptions else "",
                output_meta=OutputMeta(
                    inputs=[TextMeta(**m) for m in meta["inputs"]],
                    outputs=[
                        TextMeta(**m) if m["type"] == "text" else ImageMeta(**{k: v for k, v in m.items() if k != "type"}, extra={"provider": provider})
                        for m in meta["outputs"]
                    ],
                )
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")
