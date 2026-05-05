"""
Seedance 2.0 - BytePlus Video Generation

Professional multimodal video generation supporting text, images, video, and audio references.
Supports up to 1080p resolution. Uses BytePlus ARK SDK with async task polling.
Parameters passed as top-level request body fields.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
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
)


class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p480 = "480p"
    p720 = "720p"
    p1080 = "1080p"


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
    - Multimodal reference: provide prompt + reference_image/reference_video/reference_audio
    """

    prompt: str = Field(
        description="Text prompt describing the video content. Supports English, Japanese, Indonesian, Spanish, and Portuguese.",
        examples=["A cat stretches lazily on a sunlit windowsill, yawning as golden afternoon light filters through sheer curtains."]
    )
    image: Optional[File] = Field(
        default=None,
        description="First-frame image for image-to-video generation. Mutually exclusive with reference_image/reference_video/reference_audio."
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Last-frame image for first+last frame video generation. Requires image to be set as the first frame."
    )
    reference_image: Optional[File] = Field(
        default=None,
        description="Reference image for multimodal reference-to-video. Use prompt to describe how to use it."
    )
    reference_image_2: Optional[File] = Field(
        default=None,
        description="Second reference image for multimodal reference-to-video."
    )
    reference_image_3: Optional[File] = Field(
        default=None,
        description="Third reference image for multimodal reference-to-video."
    )
    reference_video: Optional[File] = Field(
        default=None,
        description="Reference video for multimodal generation. Max 15s, formats: mp4/mov."
    )
    reference_audio: Optional[File] = Field(
        default=None,
        description="Reference audio for multimodal generation. Max 15s, formats: wav/mp3. Requires at least one image or video."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p720,
        description="Video resolution. 1080p for highest quality, 720p for balanced, 480p for fastest."
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
        has_ref_images = any([input_data.reference_image, input_data.reference_image_2, input_data.reference_image_3])
        has_ref_video = input_data.reference_video is not None
        has_ref_audio = input_data.reference_audio is not None

        if has_ref_images or has_ref_video or has_ref_audio:
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
            for ref_img in [input_data.reference_image, input_data.reference_image_2, input_data.reference_image_3]:
                if ref_img and ref_img.exists():
                    content.append(build_image_content(ref_img.uri, role="reference_image"))

            if input_data.reference_video and input_data.reference_video.exists():
                content.append(build_video_content(input_data.reference_video.uri))

            if input_data.reference_audio:
                has_visual = any([
                    input_data.reference_image, input_data.reference_image_2,
                    input_data.reference_image_3, input_data.reference_video,
                ])
                if not has_visual:
                    raise RuntimeError("Audio reference requires at least one image or video reference.")
                if input_data.reference_audio.exists():
                    content.append(build_audio_content(input_data.reference_audio.uri))

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

            self.current_task_id = create_content_task(
                self.client,
                model=self.model_id,
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

            actual_duration = getattr(result, 'duration', float(input_data.duration) if input_data.duration > 0 else 5.0)
            fps = getattr(result, 'framespersecond', 24)
            actual_resolution = getattr(result, 'resolution', input_data.resolution.value)
            seed = getattr(result, 'seed', None)

            usage = getattr(result, 'usage', None)
            completion_tokens = None
            total_tokens = None
            if usage:
                completion_tokens = getattr(usage, 'completion_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)

            resolution_map = {
                '480p': VideoResolution.VIDEO_RES480_P,
                '720p': VideoResolution.VIDEO_RES720_P,
                '1080p': VideoResolution.VIDEO_RES1080_P,
            }
            resolution_enum = resolution_map.get(actual_resolution, VideoResolution.VIDEO_RES720_P)

            dimension_map = {
                ('480p', '16:9'): (864, 496), ('480p', '4:3'): (752, 560),
                ('480p', '1:1'): (640, 640), ('480p', '3:4'): (560, 752),
                ('480p', '9:16'): (496, 864), ('480p', '21:9'): (992, 432),
                ('720p', '16:9'): (1280, 720), ('720p', '4:3'): (1112, 834),
                ('720p', '1:1'): (960, 960), ('720p', '3:4'): (834, 1112),
                ('720p', '9:16'): (720, 1280), ('720p', '21:9'): (1470, 630),
                ('1080p', '16:9'): (1920, 1080), ('1080p', '4:3'): (1664, 1248),
                ('1080p', '1:1'): (1440, 1440), ('1080p', '3:4'): (1248, 1664),
                ('1080p', '9:16'): (1080, 1920), ('1080p', '21:9'): (2206, 946),
            }
            actual_ratio = getattr(result, 'ratio', input_data.ratio.value)
            if actual_ratio == 'adaptive':
                actual_ratio = '16:9'
            width, height = dimension_map.get((actual_resolution, actual_ratio), (1280, 720))

            estimated_tokens = int((width * height * fps * float(actual_duration)) / 1024)

            output_meta = OutputMeta(
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
                            "estimated_tokens": estimated_tokens,
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
