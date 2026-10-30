"""
P-Video-Upscale - Upscale a video to 720p, 1080p, 2k or 4k by Pruna
"""

import logging
import math
from enum import Enum
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video


class TargetEnum(str, Enum):
    r720 = "720p"
    r1080 = "1080p"
    r2k = "2k"
    r4k = "4k"


# Processed frames per target; above 30 fps only 30 frames per second are processed
FRAME_CAP = {"720p": 2400, "1080p": 1200, "2k": 600, "4k": 300}


class AppInput(BaseAppInput):
    video: File = Field(description="Video to upscale (up to 512 MiB).")
    target: TargetEnum = Field(
        default=TargetEnum.r1080,
        description="Output pixel-area target. Aspect ratio is kept; a larger source is not shrunk.",
    )
    fps: Optional[int] = Field(
        default=None,
        ge=16,
        le=60,
        description="Output frame rate (16-60). Empty keeps the source rate. Higher rates are interpolated and bill at the above-30 fps rate.",
    )
    turbo: bool = Field(default=False, description="Faster generation with slightly lower quality. Same price.")
    enable_safety_filter: bool = Field(
        default=False,
        description="Check the video against the content policy. Always-on categories are blocked either way.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Upscaled H.264 MP4 with the source audio.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-upscale"

    async def run(self, input_data: AppInput) -> AppOutput:
        source = probe_video(input_data.video.path, logger=self.logger)
        out_fps = float(input_data.fps or source["fps"] or 0)
        if source["seconds"] > 0 and out_fps > 0:
            processed = source["seconds"] * min(out_fps, 30.0)
            cap = FRAME_CAP[input_data.target.value]
            if processed > cap:
                raise ValueError(
                    f"Video too long for {input_data.target.value}: {source['seconds']:.1f}s at {out_fps:.0f} fps "
                    f"is ~{int(processed)} processed frames, the limit is {cap}. Use a shorter clip, a lower target or a lower fps."
                )

        request_data = {
            "video": file_url(input_data.video, logger=self.logger),
            "target": input_data.target.value,
            "turbo": input_data.turbo,
            "enable_safety_filter": input_data.enable_safety_filter,
        }
        if input_data.fps is not None:
            request_data["fps"] = input_data.fps

        self.logger.info(
            "p-video-upscale: source %sx%s %.2fs @ %.2f fps -> %s, fps=%s, turbo=%s",
            source["width"], source["height"], source["seconds"], source["fps"],
            input_data.target.value, input_data.fps, input_data.turbo,
        )

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        # Billed per second of the returned video, rounded up; rate doubles above 30 fps
        info = probe_video(video_path, logger=self.logger)
        seconds = info["seconds"] or source["seconds"]
        billed = float(max(1, math.ceil(seconds - 1e-3)))
        fps = info["fps"] or out_fps
        fps_tier = "high" if fps > 30.5 else "standard"

        output_meta = OutputMeta(
            inputs=[VideoMeta(width=source["width"], height=source["height"], seconds=source["seconds"], fps=int(round(source["fps"])))],
            outputs=[
                VideoMeta(
                    width=info["width"],
                    height=info["height"],
                    resolution=input_data.target.value,
                    seconds=billed,
                    fps=int(round(fps)),
                    extra={"target": input_data.target.value, "fps_tier": fps_tier},
                )
            ],
        )
        self.logger.info("Upscaled: %sx%s, %.2fs @ %.2f fps, billed %ss (%s)", info["width"], info["height"], seconds, fps, billed, fps_tier)
        return AppOutput(video=File(path=video_path), output_meta=output_meta)
