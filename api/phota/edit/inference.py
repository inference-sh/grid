import logging
from typing import List, Literal, Optional

from inferencesh import BaseApp, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import BaseModel, Field

from .phota_helper import get_api_key, phota_request, save_output_images, resolve_image_input, get_image_dimensions


class RunInput(BaseModel):
    prompt: str = Field(description="Text prompt describing the desired edit. Use [[profile_id]] to reference trained profiles.")
    images: List[File] = Field(description="Input images to edit (1-10). Accepts image files or URLs.")
    profile_ids: Optional[List[str]] = Field(None, description="Profile IDs for identity preservation. Only pass profiles relevant to this end-user.")
    num_output_images: int = Field(default=1, ge=1, le=4, description="Number of output images (1-4)")
    aspect_ratio: Literal["auto", "1:1", "3:4", "4:3", "9:16", "16:9"] = Field(default="auto", description="Output aspect ratio")
    resolution: Literal["1K", "4K"] = Field(default="1K", description="Output resolution")
    output_format: Literal["png", "jpg"] = Field(default="png", description="Output image format")


class RunOutput(BaseAppOutput):
    images: List[File] = Field(description="Edited output images")
    known_subjects: Optional[dict] = Field(None, description="Mapping of profile_id to generation count")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Phota Edit app initialized")

    async def run(self, input_data: RunInput) -> RunOutput:
        self.logger.info(f"Starting edit: {len(input_data.images)} image(s), {input_data.resolution}, {input_data.aspect_ratio}")
        image_strings = [resolve_image_input(img) for img in input_data.images]

        payload = {
            "prompt": input_data.prompt,
            "images": image_strings,
            "num_output_images": input_data.num_output_images,
            "aspect_ratio": input_data.aspect_ratio,
            "resolution": input_data.resolution,
            "output_format": input_data.output_format,
            "response_mode": "urls",
        }
        if input_data.profile_ids:
            payload["profile_ids"] = input_data.profile_ids

        result = phota_request("edit", payload, self.logger)
        self.logger.info(f"Edit complete, received {len(result.get('download_urls') or result.get('images') or [])} image(s)")

        paths = save_output_images(result, input_data.output_format, self.logger)
        output_files = [File(path=p) for p in paths]

        width, height = get_image_dimensions(paths[0])
        return RunOutput(
            images=output_files,
            known_subjects=result.get("known_subjects", {}).get("counts"),
            output_meta=OutputMeta(
                outputs=[ImageMeta(
                    width=width,
                    height=height,
                    count=len(output_files),
                    extra={"resolution": input_data.resolution},
                )]
            ),
        )
