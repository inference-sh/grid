"""
Seedance 2.0 - BytePlus Video Generation

Professional multimodal video generation supporting text, images, video, and audio references.
Supports up to 4K resolution (10-bit color). Uses BytePlus ARK SDK with async task polling.
Parameters passed as top-level request body fields.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution, ImageMeta, AudioMeta
from pydantic import Field
from typing import List, Optional
from enum import Enum
import logging

from .byteplus_helper import (
    setup_byteplus_client,
    create_content_task,
    poll_task_status,
    cancel_task,
    download_video,
    build_text_content,
    build_image_content,
    build_video_content,
    build_audio_content,
    probe_video,
)


class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p480 = "480p"
    p720 = "720p"
    p1080 = "1080p"
    p4k = "4k"


class RatioEnum(str, Enum):
    """Aspect ratio options."""
    adaptive = "adaptive"
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class AppInput(BaseAppInput):
    """Input schema for Seedance 2.0 video generation.

    Supports multiple modes:
    - Text-to-video: provide prompt only
    - Image-to-video (first frame): provide prompt + image
    - Image-to-video (first + last frame): provide prompt + image + end_image
    - Multimodal reference: provide prompt + reference_images/reference_videos/reference_audios
    """

    prompt: str = Field(
        description="Text prompt describing the video content. Supports English, Japanese, Indonesian, Spanish, and Portuguese.",
        examples=["A cat stretches lazily on a sunlit windowsill, yawning as golden afternoon light filters through sheer curtains."]
    )
    image: Optional[File] = Field(
        default=None,
        description="First-frame image for image-to-video generation. Mutually exclusive with reference inputs."
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Last-frame image for first+last frame video generation. Requires image to be set as the first frame."
    )
    reference_images: List[File] = Field(
        default=[],
        max_length=9,
        description="Reference images for multimodal generation (up to 9). Use prompt to describe how to use each, e.g. 'Image 1', 'Image 2'. Mutually exclusive with image/end_image."
    )
    reference_videos: List[File] = Field(
        default=[],
        max_length=3,
        description="Reference videos for multimodal generation (up to 3). Max 15s each, total max 15s. Formats: mp4/mov. Mutually exclusive with image/end_image."
    )
    reference_audios: List[File] = Field(
        default=[],
        max_length=3,
        description="Reference audios for multimodal generation (up to 3). Max 15s each, total max 15s. Formats: wav/mp3. Requires at least one image or video."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p720,
        description="Video resolution. 4k for ultra-high quality (10-bit color, ~2x cost of 1080p), 1080p for high quality, 720p for balanced, 480p for fastest."
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.adaptive,
        description="Aspect ratio. 'adaptive' auto-selects based on input content."
    )
    duration: int = Field(
        default=5,
        description="Duration in seconds (4-15), or -1 for auto-select."
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate synchronized audio with the video."
    )
    seed: int = Field(
        default=-1,
        description="Seed for reproducibility (-1 for random)."
    )
    watermark: bool = Field(
        default=False,
        description="Whether to add watermark to the output video."
    )
    safety_filter: bool = Field(
        default=True,
        description="Enable input safety filtering. Set to false to disable NSFW content filtering on inputs."
    )
    safety_identifier: Optional[str] = Field(
        default=None,
        description="Unique identifier of end user for platform safety policy. Must be fixed and unique per user, max 64 chars. Recommended: hash of username, user ID, or email."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 2.0 video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Seedance 2.0 video generation application using BytePlus ARK SDK."""

    async def setup(self, metadata):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = setup_byteplus_client()
        self.model_id = "dreamina-seedance-2-0-260128"
        self.unfiltered_model_id = "ep-20260514173157-ppqhm"

        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info(f"Seedance 2.0 initialized with model: {self.model_id}")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)
        return True

    def _determine_mode(self, input_data: AppInput) -> str:
        """Determine the generation mode from input."""
        has_refs = input_data.reference_images or input_data.reference_videos or input_data.reference_audios

        if has_refs:
            return "multimodal-reference"
        elif input_data.image and input_data.end_image:
            return "first-last-frame"
        elif input_data.image:
            return "image-to-video"
        else:
            return "text-to-video"

    def _build_content(self, input_data: AppInput, mode: str) -> list:
        """Build content list for BytePlus API."""
        content = []

        if input_data.prompt:
            content.append(build_text_content(input_data.prompt))

        if mode == "first-last-frame":
            if not input_data.image.exists():
                raise RuntimeError(f"First-frame image does not exist: {input_data.image.path}")
            if not input_data.end_image.exists():
                raise RuntimeError(f"Last-frame image does not exist: {input_data.end_image.path}")
            content.append(build_image_content(input_data.image.uri, role="first_frame"))
            content.append(build_image_content(input_data.end_image.uri, role="last_frame"))

        elif mode == "image-to-video":
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist: {input_data.image.path}")
            content.append(build_image_content(input_data.image.uri, role="first_frame"))

        elif mode == "multimodal-reference":
            for ref_img in input_data.reference_images:
                if ref_img.exists():
                    content.append(build_image_content(ref_img.uri, role="reference_image"))

            for ref_vid in input_data.reference_videos:
                if ref_vid.exists():
                    content.append(build_video_content(ref_vid.uri))

            if input_data.reference_audios:
                has_visual = input_data.reference_images or input_data.reference_videos
                if not has_visual:
                    raise RuntimeError("Audio reference requires at least one image or video reference.")
                for ref_aud in input_data.reference_audios:
                    if ref_aud.exists():
                        content.append(build_audio_content(ref_aud.uri))

        return content

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Seedance 2.0."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            mode = self._determine_mode(input_data)
            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Ratio: {input_data.ratio.value}, Duration: {input_data.duration}s, Audio: {input_data.generate_audio}")

            content = self._build_content(input_data, mode)

            api_params = {
                "resolution": input_data.resolution.value,
                "ratio": input_data.ratio.value,
                "duration": input_data.duration,
                "generate_audio": input_data.generate_audio,
                "seed": input_data.seed,
                "watermark": input_data.watermark,
            }
            if input_data.safety_identifier:
                api_params["safety_identifier"] = input_data.safety_identifier

            model = self.model_id if input_data.safety_filter else self.unfiltered_model_id

            self.current_task_id = create_content_task(
                self.client,
                model=model,
                content=content,
                logger=self.logger,
                **api_params,
            )

            result = await poll_task_status(
                self.client,
                self.current_task_id,
                logger=self.logger,
                poll_interval=2.0,
                cancel_flag_getter=lambda: self.cancel_flag,
            )

            video_url = None
            if hasattr(result, 'content') and hasattr(result.content, 'video_url'):
                video_url = result.content.video_url
            elif hasattr(result, 'video_url'):
                video_url = result.video_url

            if not video_url:
                self.logger.error(f"Could not extract video URL from result: {result}")
                raise RuntimeError("Failed to get video URL from response")

            video_path = download_video(video_url, self.logger)

            # Probe actual output video for real dimensions, fps, frame count
            probe = probe_video(video_path)
            width = probe.get("width", 1280)
            height = probe.get("height", 720)
            fps = probe.get("fps", 24)
            actual_duration = probe.get("seconds", float(input_data.duration) if input_data.duration > 0 else 5.0)
            actual_resolution = getattr(result, 'resolution', input_data.resolution.value)
            actual_ratio = getattr(result, 'ratio', input_data.ratio.value)
            if actual_ratio == 'adaptive':
                actual_ratio = '16:9'
            seed = getattr(result, 'seed', None)

            usage = getattr(result, 'usage', None)
            completion_tokens = None
            total_tokens = None
            if usage:
                completion_tokens = getattr(usage, 'completion_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)
            self.logger.info(f"BytePlus usage — completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, mode: {mode}, probe: {width}x{height}@{fps} {actual_duration:.3f}s")

            resolution_map = {
                '480p': VideoResolution.VIDEO_RES480_P,
                '720p': VideoResolution.VIDEO_RES720_P,
                '1080p': VideoResolution.VIDEO_RES1080_P,
                '4k': VideoResolution.VIDEO_RES4_K,
            }
            resolution_enum = resolution_map.get(actual_resolution, VideoResolution.VIDEO_RES720_P)

            # Build input metadata for pricing
            input_metas = []
            if input_data.image:
                input_metas.append(ImageMeta())
            if input_data.end_image:
                input_metas.append(ImageMeta())
            if input_data.reference_images:
                for _ in input_data.reference_images:
                    input_metas.append(ImageMeta())
            if input_data.reference_videos:
                for ref_vid in input_data.reference_videos:
                    if ref_vid and ref_vid.exists():
                        ref_probe = probe_video(ref_vid.path)
                        ref_frames = ref_probe.get("nb_frames", 0)
                        ref_fps = ref_probe.get("fps", 24)
                        ref_seconds = ref_frames / ref_fps if ref_fps > 0 else 0.0
                        input_metas.append(VideoMeta(seconds=ref_seconds))
                    else:
                        input_metas.append(VideoMeta())
            if input_data.reference_audios:
                for _ in input_data.reference_audios:
                    input_metas.append(AudioMeta())

            output_meta = OutputMeta(
                inputs=input_metas,
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_enum,
                        seconds=float(actual_duration),
                        fps=fps,
                        extra={
                            "mode": mode,
                            "ratio": actual_ratio,
                            "generate_audio": input_data.generate_audio,
                            "seed": seed,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                    )
                ]
            )

            self.logger.info(f"Video generated successfully: {video_path}")

            return AppOutput(video=File(path=video_path), output_meta=output_meta)

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
        finally:
            self.current_task_id = None
