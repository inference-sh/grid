"""
PATINA — Image to PBR Maps

Predicts seamless high-resolution PBR material maps (basecolor, normal,
roughness, metalness, height) from a single input image (photograph or render)
via fal.ai.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import List, Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_file


logging.getLogger("httpx").setLevel(logging.WARNING)


MODEL_ID = "fal-ai/patina"


class OutputFormatEnum(str, Enum):
    jpeg = "jpeg"
    png = "png"
    webp = "webp"


class AppInput(BaseAppInput):
    image: File = Field(
        description="Input image (photograph or render) to derive PBR maps from. JPEG, PNG, or WebP.",
    )
    basecolor: bool = Field(default=True, description="Predict the basecolor (albedo) map.")
    normal: bool = Field(default=True, description="Predict the normal map.")
    roughness: bool = Field(default=True, description="Predict the roughness map.")
    metalness: bool = Field(default=True, description="Predict the metalness map.")
    height: bool = Field(default=True, description="Predict the height (displacement) map.")
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output image format for the predicted maps.",
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="Enable the fal.ai safety checker on the input image.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible denoising. Leave unset for a random seed.",
    )

    def selected_maps(self) -> List[str]:
        return [
            name for name, on in [
                ("basecolor", self.basecolor),
                ("normal", self.normal),
                ("roughness", self.roughness),
                ("metalness", self.metalness),
                ("height", self.height),
            ] if on
        ]


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="Predicted PBR material map images, in the same order as map_types.")
    map_types: List[str] = Field(description="PBR map type for each image (basecolor, normal, roughness, metalness, height).")
    seed: int = Field(description="Seed used for denoising.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("PATINA image-to-maps initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "image_url": input_data.image.uri,
            "maps": input_data.selected_maps(),
            "output_format": input_data.output_format.value,
            "enable_safety_checker": input_data.enable_safety_checker,
        }
        if input_data.seed is not None and input_data.seed != -1:
            request["seed"] = input_data.seed
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        if not input_data.image.exists():
            raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

        selected = input_data.selected_maps()
        if not selected:
            raise RuntimeError("At least one map must be selected.")

        setup_fal_client()

        self.logger.info(f"Requesting {len(selected)} map(s) from {MODEL_ID}: {selected}")
        request_data = self._build_request(input_data)
        result = run_fal_model(MODEL_ID, request_data, self.logger)

        suffix = f".{input_data.output_format.value}"
        output_images: List[File] = []
        output_map_types: List[str] = []
        for i, item in enumerate(result.get("images", [])):
            url = item["url"]
            map_type = item.get("map_type") or ""
            self.logger.info(f"Downloading map {i+1} ({map_type}) from {url}")
            path = download_file(url, suffix=suffix, logger=self.logger)
            output_images.append(File(path=path))
            output_map_types.append(map_type)

        # Measure output megapixels from first map (all maps share resolution).
        width = height = 0
        if output_images:
            try:
                from PIL import Image as PILImage

                with PILImage.open(output_images[0].path) as img:
                    width, height = img.size
            except Exception as e:
                self.logger.warning(f"Could not read image dimensions: {e}")

        output_meta = OutputMeta(
            outputs=[
                ImageMeta(
                    width=width,
                    height=height,
                    count=len(output_images),
                    extra={"maps": selected},
                )
            ]
        )

        return AppOutput(
            images=output_images,
            map_types=output_map_types,
            seed=result.get("seed", 0),
            output_meta=output_meta,
        )
