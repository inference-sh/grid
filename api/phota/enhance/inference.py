import logging
from typing import List, Optional

from inferencesh import BaseApp, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .phota_helper import get_api_key, phota_request, save_base64_images, resolve_image_input, get_png_dimensions


class RunInput(BaseModel):
    image: File = Field(description="Image to enhance")
    profile_ids: Optional[List[str]] = Field(None, description="Profile IDs for identity preservation. Only pass profiles relevant to this end-user.")
    num_output_images: int = Field(default=1, ge=1, le=4, description="Number of output images (1-4)")


class RunOutput(BaseAppOutput):
    images: List[File] = Field(description="Enhanced output images")
    known_subjects: Optional[dict] = Field(None, description="Mapping of profile_id to generation count")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Phota Enhance app initialized")

    async def run(self, input_data: RunInput) -> RunOutput:
        payload = {
            "image": resolve_image_input(input_data.image),
            "num_output_images": input_data.num_output_images,
        }
        if input_data.profile_ids:
            payload["profile_ids"] = input_data.profile_ids

        result = phota_request("enhance", payload, self.logger)

        paths = save_base64_images(result["images"], self.logger)
        output_files = [File(path=p) for p in paths]

        width, height = get_png_dimensions(paths[0])
        return RunOutput(
            images=output_files,
            known_subjects=result.get("known_subjects", {}).get("counts"),
            output_meta=OutputMeta(
                outputs=[ImageMeta(
                    width=width,
                    height=height,
                    count=len(output_files),
                )]
            ),
        )
