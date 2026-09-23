"""
Grok Imagine Video 1.5 - xAI Video Generation

Text-to-video, image-to-video (first frame) and reference-to-video with xAI's
grok-imagine-video-1.5, at 480p, 720p or 1080p, with generated audio by default.
The mode follows the inputs: a first-frame image makes it image-to-video,
reference images make it reference-to-video. Editing and extension are not
supported by this model (use xai/grok-imagine-video and xai/grok-extend-video).
"""

import logging
import tempfile
from typing import List, Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution, ImageMeta
from pydantic import Field, model_validator

from .xai_helper import (
    VideoAspectRatioType,
    XAIError,
    MODERATION_NOTICE,
    is_moderated,
    upstream_cost_usd,
    create_xai_client,
    setup_logger,
    encode_image_base64,
    get_video_dimensions,
    retry_on_rate_limit,
    wait_for_video,
)

MODEL = "grok-imagine-video-1.5"

RESOLUTION_ENUM = {
    "480p": VideoResolution.VIDEO_RES480_P,
    "720p": VideoResolution.VIDEO_RES720_P,
    "1080p": VideoResolution.VIDEO_RES1080_P,
}
DIMENSIONS_1080P = {
    "16:9": (1920, 1080), "9:16": (1080, 1920), "4:3": (1440, 1080), "3:4": (1080, 1440),
    "3:2": (1620, 1080), "2:3": (1080, 1620), "1:1": (1080, 1080),
}


class AppInput(BaseAppInput):
    """Input schema for Grok Imagine Video 1.5."""

    prompt: str = Field(
        description="What happens in the video: subject, motion and camera. With reference images, refer to them as <IMAGE_0>, <IMAGE_1>, ...",
        examples=["A red fox trots through fresh snow at dawn, slow tracking shot"],
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional first frame. The video animates from this image and keeps its aspect ratio unless aspect_ratio is set.",
    )
    reference_images: List[File] = Field(
        default_factory=list,
        description="Optional reference images for characters, objects or style. Reference-to-video supports up to 720p.",
    )
    duration: int = Field(default=8, ge=1, le=15, description="Video length in seconds (1-15).")
    resolution: Literal["480p", "720p", "1080p"] = Field(
        default="480p",
        description="Output resolution. Price per second: 480p $0.08, 720p $0.14, 1080p $0.25. 1080p is not available with reference images.",
    )
    aspect_ratio: Optional[VideoAspectRatioType] = Field(
        default=None,
        description="Aspect ratio. Defaults to the first frame's ratio, or 16:9 without one. Setting it with a first frame stretches the image.",
    )
    generate_audio: bool = Field(default=True, description="Generate an audio track with the video.")

    @model_validator(mode="after")
    def _check_resolution(self):
        if self.reference_images and self.resolution == "1080p":
            raise ValueError("1080p is not available with reference images; use 480p or 720p")
        return self


class AppOutput(BaseAppOutput):
    """Output schema for Grok Imagine Video 1.5."""

    video: Optional[File] = Field(default=None, description="The generated video. Null when xAI content moderation withheld it.")
    moderated: bool = Field(default=False, description="True when xAI content moderation withheld the generated video.")
    notice: Optional[str] = Field(default=None, description="Set when xAI content moderation withheld the video.")


def _image_ref(image: File) -> str:
    if image.uri and image.uri.startswith("http"):
        return image.uri
    return encode_image_base64(image)


class App(BaseApp):
    """Grok Imagine Video 1.5 using the xAI SDK."""

    async def setup(self, metadata):
        self.logger = setup_logger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = create_xai_client()
        self._cancel = False
        self.logger.info(f"Grok Imagine Video 1.5 ready model={MODEL}")

    async def on_cancel(self):
        self._cancel = True
        return True

    def _cancelled(self) -> bool:
        ctx = getattr(self, "context", None)
        return self._cancel or bool(ctx is not None and getattr(ctx, "cancel_requested", False))

    async def run(self, input_data: AppInput) -> AppOutput:
        self._cancel = False
        try:
            if input_data.reference_images:
                mode = "reference-to-video"
            elif input_data.image:
                mode = "image-to-video"
            else:
                mode = "text-to-video"
            self.logger.info(
                f"{mode}: duration={input_data.duration}s resolution={input_data.resolution} "
                f"aspect_ratio={input_data.aspect_ratio} audio={input_data.generate_audio} "
                f"references={len(input_data.reference_images)}"
            )

            kwargs = {
                "prompt": input_data.prompt,
                "model": MODEL,
                "duration": input_data.duration,
                "resolution": input_data.resolution,
                "generate_audio": input_data.generate_audio,
            }
            if input_data.aspect_ratio:
                kwargs["aspect_ratio"] = input_data.aspect_ratio
            if input_data.image:
                kwargs["image_url"] = _image_ref(input_data.image)
            if input_data.reference_images:
                kwargs["reference_image_urls"] = [_image_ref(img) for img in input_data.reference_images]

            start = await retry_on_rate_limit(
                lambda: self.client.video.start(**kwargs), logger=self.logger, in_thread=True,
            )
            self.logger.info(f"Video request started: {start.request_id}")
            response = await wait_for_video(self.client, start.request_id, self.logger, self._cancelled)

            # A moderated video is generated and billed by xAI, so the run
            # succeeds and is billed with the video left out.
            moderated = is_moderated(response)
            cost_usd = upstream_cost_usd(response)
            video_file = None
            if moderated:
                self.logger.warning("xAI content moderation withheld the video")
            else:
                video_url = response.url
                self.logger.info(f"Downloading video from: {video_url}")
                async with httpx.AsyncClient(timeout=300) as http:
                    video_response = await http.get(video_url)
                    video_response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(video_response.content)
                    video_file = File(path=f.name)

            # xAI bills the duration it reports; fall back to the request if it reports none.
            seconds = float(response.duration or input_data.duration)
            aspect = input_data.aspect_ratio or "16:9"
            if input_data.resolution == "1080p":
                width, height = DIMENSIONS_1080P.get(aspect, (1920, 1080))
            else:
                width, height = get_video_dimensions(aspect, input_data.resolution)

            image_count = len(input_data.reference_images) + (1 if input_data.image else 0)
            output_meta = OutputMeta(
                inputs=[ImageMeta(count=image_count, extra={"type": "image_input"})] if image_count else [],
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=RESOLUTION_ENUM[input_data.resolution],
                        seconds=seconds,
                        fps=24,
                        extra={
                            "mode": mode,
                            "generate_audio": input_data.generate_audio,
                            "moderated": moderated,
                            "upstream_cost_usd": cost_usd,
                        },
                    )
                ],
            )

            self.logger.info(
                f"Video done: moderated={moderated}, billed {seconds}s at {input_data.resolution}, "
                f"{image_count} input image(s), xAI cost ${cost_usd}"
            )

            return AppOutput(
                video=video_file,
                moderated=moderated,
                notice=MODERATION_NOTICE.format(what="video") if moderated else None,
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Video generation failed: {e}")
