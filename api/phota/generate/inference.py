import logging
from typing import List, Literal, Optional

from inferencesh import BaseApp, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .phota_helper import get_api_key, phota_request, save_base64_images, get_png_dimensions


class RunInput(BaseModel):
    prompt: str = Field(description="Text prompt describing the image to generate. Use [[profile_id]] to reference trained profiles.")
    num_output_images: int = Field(default=1, ge=1, le=4, description="Number of output images (1-4)")
    aspect_ratio: Literal["auto", "1:1", "3:4", "4:3", "9:16", "16:9"] = Field(default="auto", description="Output aspect ratio")
    resolution: Literal["1K", "4K"] = Field(default="1K", description="Output resolution")


class RunOutput(BaseAppOutput):
    images: List[File] = Field(description="Generated output images")
    known_subjects: Optional[dict] = Field(None, description="Mapping of profile_id to generation count")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Phota Generate app initialized")

    async def run(self, input_data: RunInput) -> RunOutput:
        payload = {
            "prompt": input_data.prompt,
            "num_output_images": input_data.num_output_images,
            "aspect_ratio": input_data.aspect_ratio,
            "resolution": input_data.resolution,
        }

        result = phota_request("generate", payload, self.logger)

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
                    extra={"resolution": input_data.resolution},
                )]
            ),
        )
