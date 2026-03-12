"""
VACE - AI-powered video generation with character consistency
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, download_video, upload_file


class SizeEnum(str, Enum):
    portrait_720 = "720*1280"
    landscape_720 = "1280*720"
    portrait_480 = "480*832"
    landscape_480 = "832*480"


class SpeedModeEnum(str, Enum):
    lightly_juiced = "Lightly Juiced 🍊 (more consistent)"
    juiced = "Juiced 🔥 (more speed)"
    extra_juiced = "Extra Juiced 🚀 (even more speed)"


class SolverEnum(str, Enum):
    unipc = "unipc"
    dpm = "dpm++"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Description of the video content.")
    src_video: Optional[File] = Field(default=None, description="Source video to edit/transform.")
    src_mask: Optional[File] = Field(default=None, description="Mask video/image for editing.")
    src_ref_images: Optional[List[File]] = Field(default=None, description="Reference images for character consistency (1-3).")
    size: SizeEnum = Field(default=SizeEnum.landscape_480, description="Output video resolution.")
    frame_num: int = Field(default=81, description="Number of frames to generate.")
    speed_mode: SpeedModeEnum = Field(default=SpeedModeEnum.lightly_juiced, description="Speed vs quality balance.")
    sample_steps: int = Field(default=50, description="Sampling steps (higher = better quality).")
    sample_solver: SolverEnum = Field(default=SolverEnum.unipc, description="Sampling algorithm.")
    sample_guide_scale: float = Field(default=5.0, description="How closely to follow the prompt.")
    sample_shift: int = Field(default=16, description="Sample shift parameter.")
    seed: int = Field(default=-1, description="Random seed (-1 for random).")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file.")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "vace"

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        try:
            self.logger.info(f"Generating video: {input_data.prompt[:100]}...")
            request_data = {
                "prompt": input_data.prompt,
                "size": input_data.size.value,
                "frame_num": input_data.frame_num,
                "speed_mode": input_data.speed_mode.value,
                "sample_steps": input_data.sample_steps,
                "sample_solver": input_data.sample_solver.value,
                "sample_guide_scale": input_data.sample_guide_scale,
                "sample_shift": input_data.sample_shift,
                "seed": input_data.seed,
            }
            
            if input_data.src_video:
                if input_data.src_video.uri and input_data.src_video.uri.startswith("http"):
                    request_data["src_video"] = input_data.src_video.uri
                else:
                    upload_result = upload_file(input_data.src_video.path, logger=self.logger)
                    request_data["src_video"] = upload_result.get("urls", {}).get("get")

            if input_data.src_mask:
                if input_data.src_mask.uri and input_data.src_mask.uri.startswith("http"):
                    request_data["src_mask"] = input_data.src_mask.uri
                else:
                    upload_result = upload_file(input_data.src_mask.path, logger=self.logger)
                    request_data["src_mask"] = upload_result.get("urls", {}).get("get")

            if input_data.src_ref_images:
                ref_urls = []
                for img in input_data.src_ref_images:
                    if img.uri and img.uri.startswith("http"):
                        ref_urls.append(img.uri)
                    else:
                        upload_result = upload_file(img.path, logger=self.logger)
                        ref_urls.append(upload_result.get("urls", {}).get("get"))
                request_data["src_ref_images"] = ref_urls

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=False, logger=self.logger)
            generation_url = result.get("generation_url")
            if not generation_url:
                raise RuntimeError("No generation_url in response")
            if generation_url.startswith("/"):
                generation_url = f"https://api.pruna.ai{generation_url}"

            video_path = download_video(generation_url, logger=self.logger)
            
            # Parse dimensions from size
            dims = input_data.size.value.split("*")
            width, height = int(dims[0]), int(dims[1])

            # Determine resolution from size
            is_720p = "720" in input_data.size.value
            resolution = VideoResolution.VIDEO_RES720_P if is_720p else VideoResolution.VIDEO_RES480_P

            return AppOutput(video=File(path=video_path), output_meta=OutputMeta(outputs=[VideoMeta(width=width, height=height, resolution=resolution, seconds=float(input_data.frame_num / 24))]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
