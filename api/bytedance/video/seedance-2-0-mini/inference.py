"""
Seedance 2.0 Mini - BytePlus Video Generation

Cost-effective multimodal video generation supporting text, images, video, and audio references.
~50% cheaper than Seedance 2.0 with multimodal references, video editing, and extension.
Uses BytePlus ARK SDK with async task polling. Parameters passed as top-level request body fields.
"""

from inferencesh import BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Any, ClassVar, List, Optional
from enum import Enum

from .seedance_base import SeedanceApp


class ResolutionEnum(str, Enum):
    """Video resolution options (Seedance 2.0 Mini supports 480p and 720p only)."""
    p480 = "480p"
    p720 = "720p"


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
    """Input schema for Seedance 2.0 Mini video generation.

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
        description="Video resolution. Seedance 2.0 Mini supports 480p and 720p only."
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
    """Output schema for Seedance 2.0 Mini video generation."""

    video: File = Field(description="The generated video file.")


class App(SeedanceApp):
    """Seedance 2.0 Mini video generation application using BytePlus ARK SDK."""

    display_name: ClassVar[str] = "Seedance 2.0 Mini"
    model_id: ClassVar[str] = "dreamina-seedance-2-0-mini-260615"
    # No unfiltered endpoint: safety_filter=False still uses model_id.
    unfiltered_model_id: ClassVar[Optional[str]] = None
    OutputType: ClassVar[Any] = AppOutput

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Seedance 2.0 Mini."""
        return await super().run(input_data, metadata)
