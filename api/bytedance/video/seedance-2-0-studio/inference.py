"""
Seedance 2.0 Studio - BytePlus Video Generation with Asset Library

Same capabilities as Seedance 2.0, but automatically uploads reference images
to BytePlus private virtual portrait library for enhanced character consistency.
Uses asset:// URIs instead of direct image URLs for trusted asset generation.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
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
)
from .asset_library_helper import (
    setup_asset_client,
    create_asset_group,
    upload_and_activate,
)


# --- Shared constants ---

DIMENSION_MAP = {
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

RESOLUTION_MAP = {
    '480p': VideoResolution.VIDEO_RES480_P,
    '720p': VideoResolution.VIDEO_RES720_P,
    '1080p': VideoResolution.VIDEO_RES1080_P,
}


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
    """Input schema for Seedance 2.0 Studio video generation.

    Same as Seedance 2.0, but images are automatically uploaded to the
    private virtual portrait library for enhanced character consistency.

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
    safety_filter: bool = Field(
        default=True,
        description="Enable input safety filtering. Set to false to disable NSFW content filtering on inputs."
    )
    safety_identifier: Optional[str] = Field(
        default=None,
        description="Unique identifier of end user for platform safety policy. Must be fixed and unique per user, max 64 chars. Recommended: hash of username, user ID, or email. Also used to namespace asset groups."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 2.0 Studio video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Seedance 2.0 Studio video generation with private asset library."""

    async def setup(self, metadata):
        """Initialize BytePlus clients for generation and asset library."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = setup_byteplus_client()
        self.asset_client = setup_asset_client()
        self.model_id = "dreamina-seedance-2-0-260128"
        self.unfiltered_model_id = "ep-20260514173157-ppqhm"
        self.asset_group_id = None

        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info(f"Seedance 2.0 Studio initialized with model: {self.model_id}")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)
        return True

    async def _ensure_asset_group(self, safety_identifier: Optional[str] = None):
        """Create asset group, namespaced by safety_identifier if provided."""
        if self.asset_group_id is None:
            group_name = f"seedance-studio-{safety_identifier}" if safety_identifier else "seedance-studio-assets"
            self.asset_group_id = create_asset_group(
                self.asset_client,
                name=group_name,
                description="Auto-managed asset group for Seedance 2.0 Studio",
                logger=self.logger,
            )
        return self.asset_group_id

    async def _upload_image_asset(self, file: File) -> str:
        """Upload a single image to asset library, return asset:// URI."""
        if not file or not file.exists():
            raise RuntimeError(f"Image file does not exist: {file}")
        group_id = await self._ensure_asset_group()
        asset_uri = await upload_and_activate(
            self.asset_client,
            group_id,
            file.uri,
            asset_type="Image",
            logger=self.logger,
        )
        return asset_uri

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

    async def _build_content(self, input_data: AppInput, mode: str) -> list:
        """Build content list, uploading images to asset library first."""
        content = []

        if input_data.prompt:
            content.append(build_text_content(input_data.prompt))

        if mode == "first-last-frame":
            first_uri = await self._upload_image_asset(input_data.image)
            last_uri = await self._upload_image_asset(input_data.end_image)
            content.append(build_image_content(first_uri, role="first_frame"))
            content.append(build_image_content(last_uri, role="last_frame"))

        elif mode == "image-to-video":
            first_uri = await self._upload_image_asset(input_data.image)
            content.append(build_image_content(first_uri, role="first_frame"))

        elif mode == "multimodal-reference":
            for ref_img in input_data.reference_images:
                if ref_img.exists():
                    asset_uri = await self._upload_image_asset(ref_img)
                    content.append(build_image_content(asset_uri, role="reference_image"))

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

    def _build_output_meta(self, input_data: AppInput, result, mode: str) -> OutputMeta:
        """Build output metadata from generation result."""
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

        resolution_enum = RESOLUTION_MAP.get(actual_resolution, VideoResolution.VIDEO_RES720_P)

        actual_ratio = getattr(result, 'ratio', input_data.ratio.value)
        if actual_ratio == 'adaptive':
            actual_ratio = '16:9'
        width, height = DIMENSION_MAP.get((actual_resolution, actual_ratio), (1280, 720))

        estimated_tokens = int((width * height * fps * float(actual_duration)) / 1024)

        return OutputMeta(
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
                        "studio": True,
                    }
                )
            ]
        )

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Seedance 2.0 with asset library."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            mode = self._determine_mode(input_data)
            self.logger.info(f"Starting {mode} generation (studio)")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Ratio: {input_data.ratio.value}, Duration: {input_data.duration}s, Audio: {input_data.generate_audio}")

            # Ensure asset group is namespaced by safety_identifier
            await self._ensure_asset_group(input_data.safety_identifier)

            content = await self._build_content(input_data, mode)

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
            output_meta = self._build_output_meta(input_data, result, mode)

            self.logger.info(f"Video generated successfully: {video_path}")

            return AppOutput(video=File(path=video_path), output_meta=output_meta)

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
        finally:
            self.current_task_id = None
