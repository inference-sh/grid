"""
Seedance 1.5 Pro Text-to-Video

Generate videos with audio from text prompts using ByteDance's Seedance 1.5 Pro model.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

# Suppress noisy httpx polling logs at module level
logging.getLogger("httpx").setLevel(logging.WARNING)


class AspectRatioEnum(str, Enum):
    """Video aspect ratio options."""
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_4_3 = "4:3"
    ratio_1_1 = "1:1"
    ratio_3_4 = "3:4"
    ratio_9_16 = "9:16"


class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p480 = "480p"
    p720 = "720p"


class DurationEnum(str, Enum):
    """Video duration options in seconds."""
    s4 = "4"
    s5 = "5"
    s6 = "6"
    s7 = "7"
    s8 = "8"
    s9 = "9"
    s10 = "10"
    s11 = "11"
    s12 = "12"


# Video dimensions lookup table: (resolution, aspect_ratio) -> (width, height)
VIDEO_DIMENSIONS = {
    ("480p", "21:9"): (1120, 480),
    ("480p", "16:9"): (854, 480),
    ("480p", "4:3"): (640, 480),
    ("480p", "1:1"): (480, 480),
    ("480p", "3:4"): (480, 640),
    ("480p", "9:16"): (480, 854),
    ("720p", "21:9"): (1680, 720),
    ("720p", "16:9"): (1280, 720),
    ("720p", "4:3"): (960, 720),
    ("720p", "1:1"): (720, 720),
    ("720p", "3:4"): (720, 960),
    ("720p", "9:16"): (720, 1280),
}


class AppInput(BaseAppInput):
    """Input schema for Seedance 1.5 Pro Text-to-Video."""
    
    prompt: str = Field(
        description="The text prompt used to generate the video. Describe the scene, subject, and motion.",
        examples=["A cat walking gracefully across a sunlit garden, soft morning light filtering through leaves"]
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="The aspect ratio of the generated video."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p720,
        description="Video resolution. 480p for faster generation, 720p for better quality."
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (4-12 seconds)."
    )
    camera_fixed: bool = Field(
        default=False,
        description="Whether to fix the camera position during video generation."
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio for the video."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation. Use -1 or None for random."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 1.5 Pro Text-to-Video."""
    
    video: File = Field(description="The generated video file with optional audio.")
    seed: int = Field(description="The seed used for generation.")


class App(BaseApp):
    """Seedance 1.5 Pro Text-to-Video application."""
    
    async def setup(self, metadata):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        
        # fal.ai model endpoint for text-to-video
        self.model_id = "fal-ai/bytedance/seedance/v1.5/pro/text-to-video"
        
        self.logger.info(f"Seedance 1.5 Pro T2V initialized with model: {self.model_id}")

    def _build_request(self, input_data: AppInput) -> dict:
        """Build the request payload for fal.ai API."""
        request = {
            "prompt": input_data.prompt,
            "aspect_ratio": input_data.aspect_ratio.value,
            "resolution": input_data.resolution.value,
            "duration": input_data.duration.value,
            "camera_fixed": input_data.camera_fixed,
            "generate_audio": input_data.generate_audio,
        }
        
        # Optional seed
        if input_data.seed is not None and input_data.seed != -1:
            request["seed"] = input_data.seed
        
        return request

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from text prompt using Seedance 1.5 Pro."""
        try:
            # Setup fal.ai client
            setup_fal_client()
            
            self.logger.info(f"Generating video with prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Settings: {input_data.resolution.value}, {input_data.aspect_ratio.value}, {input_data.duration.value}s")
            
            # Build and send request
            request_data = self._build_request(input_data)
            result = run_fal_model(self.model_id, request_data, self.logger)
            
            # Download generated video
            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)
            
            # Build output metadata for pricing
            resolution_map = {
                "480p": VideoResolution.RES_480P,
                "720p": VideoResolution.RES_720P,
            }
            width, height = VIDEO_DIMENSIONS.get(
                (input_data.resolution.value, input_data.aspect_ratio.value),
                (1280, 720)  # default fallback
            )
            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_map.get(input_data.resolution.value),
                        seconds=float(input_data.duration.value),
                        fps=24,
                        extra={"generate_audio": input_data.generate_audio}
                    )
                ]
            )
            
            return AppOutput(
                video=File(path=video_path),
                seed=result["seed"],
                output_meta=output_meta
            )
            
        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
