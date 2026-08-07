"""
Seedance 2.0 - BytePlus Video Generation

Professional multimodal video generation supporting text, images, video, and audio references.
Supports up to 4K resolution (10-bit color). Uses BytePlus ARK SDK with async task polling.
Parameters passed as top-level request body fields.
"""

from inferencesh import BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Any, ClassVar, List, Optional
from enum import Enum

from .seedance_base import SeedanceApp


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
        description="Text prompt describing the video content. Use @Image1, @Image2, @Video1, @Audio1 to reference inputs in order (e.g. '@Image1 is the style reference, @Video1 provides the motion'). Supports English, Japanese, Indonesian, Spanish, and Portuguese.",
        examples=["@Image1 is the character reference. @Video1 provides the motion. A cat stretches lazily on a sunlit windowsill."]
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
    safety_identifier: Optional[str] = Field(
        default=None,
        description="Unique identifier of end user for platform safety policy. Must be fixed and unique per user, max 64 chars. Recommended: hash of username, user ID, or email."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 2.0 video generation."""

    video: File = Field(description="The generated video file.")


class App(SeedanceApp):
    """Seedance 2.0 video generation application using BytePlus ARK SDK."""

    display_name: ClassVar[str] = "Seedance 2.0"
    model_id: ClassVar[str] = "dreamina-seedance-2-0-260128"
    OutputType: ClassVar[Any] = AppOutput

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Seedance 2.0."""
        return await super().run(input_data, metadata)
