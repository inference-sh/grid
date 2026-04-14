"""
PATINA Extract Material — Image-to-Material Extraction

Extracts a seamlessly tiling texture (plus PBR material maps) from a region
of a source image described by a text prompt, via fal.ai PATINA.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import List, Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_file


logging.getLogger("httpx").setLevel(logging.WARNING)


MODEL_ID = "fal-ai/patina/material/extract"


class OutputFormatEnum(str, Enum):
    jpeg = "jpeg"
    png = "png"
    webp = "webp"


class ImageSizeEnum(str, Enum):
    square_hd = "square_hd"
    square = "square"
    portrait_4_3 = "portrait_4_3"
    portrait_16_9 = "portrait_16_9"
    landscape_4_3 = "landscape_4_3"
    landscape_16_9 = "landscape_16_9"


class TilingModeEnum(str, Enum):
    both = "both"
    horizontal = "horizontal"
    vertical = "vertical"


class UpscaleFactorEnum(int, Enum):
    none = 0
    x2 = 2
    x4 = 4


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Describe which texture to extract from the image.",
        examples=["the wall"],
    )
    image: File = Field(
        description="Image to extract a texture from. JPEG, PNG, or WebP.",
    )
    image_size: ImageSizeEnum = Field(
        default=ImageSizeEnum.square_hd,
        description="Output texture dimensions preset.",
    )
    num_inference_steps: int = Field(
        default=8, ge=1, le=8,
        description="Number of denoising steps for texture generation.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )
    num_images: int = Field(
        default=1, ge=1, le=4,
        description="Number of texture images to generate.",
    )
    enable_prompt_expansion: bool = Field(
        default=True,
        description="Expand prompt with an LLM for richer detail (adds a small cost).",
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="Enable the safety checker for generated images.",
    )
    tiling_mode: TilingModeEnum = Field(
        default=TilingModeEnum.both,
        description="Tiling direction: 'both' (omnidirectional), 'horizontal', or 'vertical'.",
    )
    tile_size: int = Field(
        default=128, ge=32, le=256,
        description="Tile size in latent space (64 = 512px, 128 = 1024px).",
    )
    tile_stride: int = Field(
        default=64, ge=16, le=128,
        description="Tile stride in latent space.",
    )
    strength: float = Field(
        default=0.75, gt=0.0, le=1.0,
        description="How much to transform the input image during extraction.",
    )
    basecolor: bool = Field(default=True, description="Predict the basecolor (albedo) map.")
    normal: bool = Field(default=True, description="Predict the normal map.")
    roughness: bool = Field(default=True, description="Predict the roughness map.")
    metalness: bool = Field(default=True, description="Predict the metalness map.")
    height: bool = Field(default=True, description="Predict the height (displacement) map.")
    upscale_factor: UpscaleFactorEnum = Field(
        default=UpscaleFactorEnum.none,
        description="Upscale factor for PBR maps via SeedVR seamless upscaling. Base texture is not upscaled.",
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output image format for textures and PBR maps.",
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
    images: List[File] = Field(
        description="Extracted tileable texture and predicted PBR material maps, in the same order as map_types.",
    )
    map_types: List[str] = Field(
        description="Tag for each image: empty string for base texture, otherwise the PBR map type (basecolor, normal, roughness, metalness, height).",
    )
    seed: int = Field(description="Seed used for texture generation.")
    prompt: str = Field(description="The prompt used for generation (possibly expanded).")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("PATINA extract-material initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "image_url": input_data.image.uri,
            "image_size": input_data.image_size.value,
            "num_inference_steps": input_data.num_inference_steps,
            "num_images": input_data.num_images,
            "enable_prompt_expansion": input_data.enable_prompt_expansion,
            "enable_safety_checker": input_data.enable_safety_checker,
            "tiling_mode": input_data.tiling_mode.value,
            "tile_size": input_data.tile_size,
            "tile_stride": input_data.tile_stride,
            "strength": input_data.strength,
            "maps": input_data.selected_maps(),
            "upscale_factor": int(input_data.upscale_factor.value),
            "output_format": input_data.output_format.value,
        }
        if input_data.seed is not None and input_data.seed != -1:
            request["seed"] = input_data.seed
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        if not input_data.image.exists():
            raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

        setup_fal_client()

        selected = input_data.selected_maps()
        self.logger.info(
            f"Extracting '{input_data.prompt[:80]}' texture "
            f"({input_data.num_images} image(s), maps={selected})"
        )
        request_data = self._build_request(input_data)
        result = run_fal_model(MODEL_ID, request_data, self.logger)

        suffix = f".{input_data.output_format.value}"
        output_images: List[File] = []
        output_map_types: List[str] = []
        for i, item in enumerate(result.get("images", [])):
            url = item["url"]
            map_type = item.get("map_type") or ""
            self.logger.info(f"Downloading image {i+1} (map_type={map_type}) from {url}")
            path = download_file(url, suffix=suffix, logger=self.logger)
            output_images.append(File(path=path))
            output_map_types.append(map_type)

        base_path = None
        for p, mt in zip(output_images, output_map_types):
            if mt == "":
                base_path = p.path
                break
        if base_path is None and output_images:
            base_path = output_images[0].path

        width = height = 0
        if base_path:
            try:
                from PIL import Image as PILImage

                with PILImage.open(base_path) as img:
                    width, height = img.size
            except Exception as e:
                self.logger.warning(f"Could not read image dimensions: {e}")

        output_meta = OutputMeta(
            outputs=[
                ImageMeta(
                    width=width,
                    height=height,
                    count=len(output_images),
                    extra={
                        "num_texture_images": input_data.num_images,
                        "maps": selected,
                        "upscale_factor": int(input_data.upscale_factor.value),
                    },
                )
            ]
        )

        return AppOutput(
            images=output_images,
            map_types=output_map_types,
            seed=result.get("seed", 0),
            prompt=result.get("prompt", input_data.prompt),
            output_meta=output_meta,
        )
