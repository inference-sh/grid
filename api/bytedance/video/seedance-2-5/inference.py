"""
Seedance 2.5 - BytePlus Video Generation

Professional multimodal video generation supporting text, images, video, and audio references.
Supports 480p/720p resolution, durations up to 30s, and MOV output format.
Uses BytePlus ARK SDK with async task polling.
"""

from inferencesh import BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Any, ClassVar, List, Optional
from enum import Enum

from .seedance_base import SeedanceApp


class ResolutionEnum(str, Enum):
    p480 = "480p"
    p720 = "720p"


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
    """Input schema for Seedance 2.5 video generation.

    Three task categories (auto-detected from parameters):
    - First frame / First & last frame: set image (and optionally end_image)
    - Multi-reference: set reference_images / reference_videos / reference_audios (audio-only supported)
    - Text-to-video: provide prompt only

    The API determines the specific task type (Video Editing, Video Extension,
    or Video Reference) automatically. For Video Editing tasks, duration must
    be -1. For First & Last Frame, Video Editing, Video Extension, and
    first-frame image-to-video tasks, ratio must be 'adaptive'.
    """

    prompt: str = Field(
        description="Text prompt describing the video content. Supports English, Spanish, Indonesian, Portuguese, Japanese, Malay, Thai, Arabic, Vietnamese, and Korean.",
        examples=["A cat stretches lazily on a sunlit windowsill, yawning as golden afternoon light filters through sheer curtains."]
    )
    image: Optional[File] = Field(
        default=None,
        description="First-frame image for image-to-video generation. Ratio must be 'adaptive'. Mutually exclusive with reference inputs."
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Last-frame image for first+last frame video generation. Requires image to be set as the first frame. Ratio must be 'adaptive'."
    )
    reference_images: List[File] = Field(
        default=[],
        max_length=30,
        description="Reference images for multimodal generation (up to 30). Use prompt to describe how to use each, e.g. 'Image 1', 'Image 2'. Mutually exclusive with image/end_image."
    )
    reference_videos: List[File] = Field(
        default=[],
        max_length=10,
        description="Reference videos for multimodal generation (up to 10). Each 2-30s, total max 30s. Formats: mp4/mov (MOV recommended for best quality). Mutually exclusive with image/end_image."
    )
    reference_audios: List[File] = Field(
        default=[],
        max_length=10,
        description="Reference audios for multimodal generation (up to 10). Each 2-30s, total max 30s. Formats: wav/mp3. Audio-only input is supported."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p720,
        description="Video resolution. 720p for balanced quality, 480p for fastest."
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.adaptive,
        description="Aspect ratio. Must be 'adaptive' for image-to-video, first+last frame, video editing, and video extension tasks. 'adaptive' auto-selects based on input content."
    )
    duration: int = Field(
        default=-1,
        description="Duration in seconds (4-30), or -1 for auto-select. Must be -1 for Video Editing tasks."
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.mp4,
        description="Output video format. MOV provides higher color precision (YUV 4:4:4); recommended for professional workflows and when using video references."
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
    safety_identifier: Optional[str] = Field(
        default=None,
        description="Unique identifier of end user for platform safety policy. Must be fixed and unique per user, max 64 chars. Recommended: hash of username, user ID, or email."
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")


class App(SeedanceApp):
    display_name: ClassVar[str] = "Seedance 2.5"
    model_id: ClassVar[str] = "dreamina-seedance-2-5-260628"
    supports_audio_only: ClassVar[bool] = True
    OutputType: ClassVar[Any] = AppOutput

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        return await super().run(input_data, metadata)
