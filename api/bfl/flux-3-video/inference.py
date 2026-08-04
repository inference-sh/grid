import os
import logging
from typing import Optional, List, Union
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field

from .bfl_helper import BFLClient, download_file


class ResolutionEnum(str, Enum):
    hd = "hd"
    fhd = "fhd"


class AspectRatioEnum(str, Enum):
    auto = "auto"
    r21_9 = "21:9"
    r2_1 = "2:1"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


RESOLUTION_DIMS = {
    ("hd", "21:9"): (1440, 608),
    ("hd", "2:1"): (1440, 704),
    ("hd", "16:9"): (1280, 704),
    ("hd", "4:3"): (1024, 768),
    ("hd", "1:1"): (960, 960),
    ("hd", "3:4"): (768, 1024),
    ("hd", "9:16"): (704, 1280),
    ("fhd", "21:9"): (2176, 928),
    ("fhd", "2:1"): (2048, 1024),
    ("fhd", "16:9"): (1920, 1088),
    ("fhd", "4:3"): (1536, 1152),
    ("fhd", "1:1"): (1440, 1440),
    ("fhd", "3:4"): (1152, 1536),
    ("fhd", "9:16"): (1088, 1920),
}


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt describing the video to generate.",
    )
    image: Optional[File] = Field(
        default=None,
        description="Single keyframe image. The image becomes the opening frame of the video (image-to-video).",
    )
    keyframes: Optional[List] = Field(
        default=None,
        description="Multiple keyframes as [seconds, image_url] pairs for storyboard-style generation. Overrides image if both provided.",
    )
    start_video: Optional[File] = Field(
        default=None,
        description="Video to continue from (video-to-video continuation).",
    )
    draft_cache: Optional[str] = Field(
        default=None,
        description="Draft cache bundle from a prior draft generation for full-quality render (draft_enhance).",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.hd,
        description="Output resolution. hd: up to 1MP per frame. fhd: up to 2MP per frame. Draft mode only supports hd.",
    )
    duration: Optional[int] = Field(
        default=None,
        ge=5,
        le=20,
        description="Video duration in seconds (5-20). Default: auto.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.auto,
        description="Output aspect ratio.",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate synchronized audio. Included at no extra charge.",
    )
    draft: bool = Field(
        default=False,
        description="Generate a fast HD preview. Result includes a draft_cache for later full-quality render.",
    )
    safety_tolerance: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="Safety filter tolerance: 0 (strictest) to 4. Default: 2.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file.")
    draft_cache: Optional[str] = Field(
        default=None,
        description="Draft cache bundle for full-quality render. Only present when draft=true.",
    )


def _detect_mode(input_data: AppInput) -> str:
    if input_data.draft_cache:
        return "draft_enhance"
    if input_data.start_video:
        return "v2v"
    if input_data.keyframes or input_data.image:
        return "i2v"
    return "t2v"


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("BFL_KEY")
        if not api_key:
            raise RuntimeError("BFL_KEY must be set")
        self.client = BFLClient(api_key=api_key, logger=self.logger)
        self.logger.info("BFL FLUX 3 Video initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        mode = _detect_mode(input_data)
        self.logger.info(
            f"Mode: {mode}, resolution: {input_data.resolution.value}, "
            f"aspect_ratio: {input_data.aspect_ratio.value}, draft: {input_data.draft}"
        )

        payload = {
            "mode": mode,
            "prompt": input_data.prompt,
            "resolution": input_data.resolution.value,
            "generate_audio": input_data.generate_audio,
        }

        if input_data.duration is not None:
            payload["duration"] = input_data.duration
        if input_data.aspect_ratio != AspectRatioEnum.auto:
            payload["aspect_ratio"] = input_data.aspect_ratio.value
        if input_data.safety_tolerance is not None:
            payload["safety_tolerance"] = input_data.safety_tolerance
        if input_data.draft:
            payload["draft"] = True

        input_metas = []

        if mode == "i2v":
            if input_data.keyframes:
                resolved = []
                for item in input_data.keyframes:
                    if isinstance(item, list) and len(item) == 2:
                        ts, img = item
                        uri = img.uri if isinstance(img, File) else img
                        resolved.append([ts, uri])
                        input_metas.append(ImageMeta())
                    elif isinstance(item, (str, File)):
                        uri = item.uri if isinstance(item, File) else item
                        resolved.append(uri)
                        input_metas.append(ImageMeta())
                    else:
                        resolved.append(item)
                payload["keyframes"] = resolved
            elif input_data.image:
                payload["keyframes"] = input_data.image.uri
                input_metas.append(ImageMeta())

        if mode == "v2v" and input_data.start_video:
            payload["start_video"] = input_data.start_video.uri

        if mode == "draft_enhance" and input_data.draft_cache:
            payload["draft_cache"] = input_data.draft_cache

        result = await self.client.submit_and_poll("/v1/flux-3-video", payload)

        if not result.sample_url:
            raise RuntimeError("No video URL in completed result")

        self.logger.info(f"Video ready: {result.sample_url[:80]}...")
        video_path = await download_file(result.sample_url, suffix=".mp4", logger=self.logger)

        ar = input_data.aspect_ratio.value
        if ar == "auto":
            ar = "16:9"
        res = input_data.resolution.value
        w, h = RESOLUTION_DIMS.get((res, ar), (1280, 704))

        duration = input_data.duration or 5

        return AppOutput(
            video=File(path=video_path),
            draft_cache=result.draft_cache,
            output_meta=OutputMeta(
                inputs=input_metas,
                outputs=[VideoMeta(
                    width=w, height=h,
                    seconds=float(duration),
                    extra={
                        "model": "flux-3-video",
                        "mode": mode,
                        "resolution": res,
                        "draft": input_data.draft,
                    },
                )],
            ),
        )

    async def unload(self):
        await self.client.close()
