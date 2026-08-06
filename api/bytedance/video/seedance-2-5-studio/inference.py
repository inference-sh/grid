"""
Seedance 2.5 Studio - BytePlus Video Generation with Asset Library

Same capabilities as Seedance 2.5, but automatically uploads reference images,
videos, and audio to the BytePlus private virtual portrait library for enhanced
character consistency. Uses asset:// URIs instead of direct URLs for trusted
asset generation.
"""

from inferencesh import BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Any, ClassVar, List, Optional
from enum import Enum

from .seedance_base import SeedanceStudioApp


class ResolutionEnum(str, Enum):
    p480 = "480p"
    p720 = "720p"
    p1080 = "1080p"
    p4k = "4k"


class RatioEnum(str, Enum):
    adaptive = "adaptive"
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class OutputFormatEnum(str, Enum):
    mp4 = "mp4"
    mov = "mov"


class AppInput(BaseAppInput):
    """Input schema for Seedance 2.5 Studio video generation.

    Same as Seedance 2.5, but all references are automatically uploaded to the
    private virtual portrait library for enhanced character consistency.

    Three task categories (auto-detected from parameters):
    - First frame / First & last frame: set image (and optionally end_image)
    - Multi-reference: set reference_images / reference_videos / reference_audios
    - Text-to-video: provide prompt only

    The API determines the specific task type (Video Editing, Video Extension,
    or Video Reference) automatically. For Video Editing tasks, duration is
    forced to -1. For First & Last Frame, Video Editing, or Video Extension
    tasks, ratio must be 'adaptive'.
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
        description="Last-frame image for first+last frame video generation. Requires image to be set as the first frame. Ratio must be 'adaptive'."
    )
    reference_images: List[File] = Field(
        default=[],
        max_length=9,
        description="Reference images for multimodal generation (up to 9). Use prompt to describe how to use each, e.g. 'Image 1', 'Image 2'. Mutually exclusive with image/end_image."
    )
    reference_videos: List[File] = Field(
        default=[],
        max_length=3,
        description="Reference videos for multimodal generation (up to 3). Max 15s each, total max 15s. Formats: mp4/mov (MOV recommended for best quality). Mutually exclusive with image/end_image."
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
        description="Aspect ratio. Must be 'adaptive' for first+last frame, video editing, and video extension tasks. 'adaptive' auto-selects based on input content."
    )
    duration: int = Field(
        default=5,
        description="Duration in seconds (4-30). Set to -1 for Video Editing tasks (auto-determined by API)."
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.mp4,
        description="Output video format. MOV provides higher quality; recommended when using video references."
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
    video: File = Field(description="The generated video file.")


class App(SeedanceStudioApp):
    display_name: ClassVar[str] = "Seedance 2.5 Studio"
    model_id: ClassVar[str] = "dreamina-seedance-2-5-260628"
    unfiltered_model_id: ClassVar[Optional[str]] = None
    OutputType: ClassVar[Any] = AppOutput

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        return await super().run(input_data, metadata)
