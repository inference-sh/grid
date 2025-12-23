from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import ClassVar, Optional
from enum import Enum
import tempfile
import os
import logging
import httpx

from kling import KlingClient


class ModeEnum(str, Enum):
    """Video generation mode."""
    std = "std"  # Standard mode - cost effective
    pro = "pro"  # Professional mode - higher quality


class AspectRatioEnum(str, Enum):
    """Aspect ratio options."""
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class DurationEnum(str, Enum):
    """Video duration options."""
    seconds_5 = "5"
    seconds_10 = "10"


class CameraTypeEnum(str, Enum):
    """Predefined camera movement types."""
    none = "none"  # No camera control
    simple = "simple"  # Custom camera movement via config
    down_back = "down_back"  # Camera descends and moves backward
    forward_up = "forward_up"  # Camera moves forward and tilts up
    right_turn_forward = "right_turn_forward"  # Rotate right and move forward
    left_turn_forward = "left_turn_forward"  # Rotate left and move forward


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Positive text prompt for video generation. Cannot exceed 2500 characters."
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Negative text prompt. Cannot exceed 2500 characters."
    )
    mode: ModeEnum = Field(
        ModeEnum.std,
        description="Video generation mode. 'std' is cost-effective, 'pro' provides higher quality."
    )
    aspect_ratio: AspectRatioEnum = Field(
        AspectRatioEnum.ratio_16_9,
        description="Aspect ratio of the generated video (width:height)."
    )
    duration: DurationEnum = Field(
        DurationEnum.seconds_5,
        description="Video length in seconds."
    )
    cfg_scale: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Flexibility in video generation. Higher value = lower flexibility, stronger relevance to prompt."
    )
    camera_type: CameraTypeEnum = Field(
        CameraTypeEnum.none,
        description="Predefined camera movement type."
    )
    # Camera config (only used when camera_type is 'simple')
    camera_horizontal: Optional[float] = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera horizontal movement. Negative=left, positive=right. Only one camera config should be non-zero."
    )
    camera_vertical: Optional[float] = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera vertical movement. Negative=down, positive=up."
    )
    camera_pan: Optional[float] = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera pan (rotation around x-axis). Negative=down, positive=up."
    )
    camera_tilt: Optional[float] = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera tilt (rotation around y-axis). Negative=left, positive=right."
    )
    camera_roll: Optional[float] = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera roll (rotation around z-axis). Negative=counterclockwise, positive=clockwise."
    )
    camera_zoom: Optional[float] = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera zoom. Negative=zoom in (narrower FOV), positive=zoom out (wider FOV)."
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")
    video_id: str = Field(description="Kling video ID for potential video extension")
    duration: str = Field(description="Video duration in seconds")


class App(BaseApp):
    # Model name for this variant
    MODEL_NAME: ClassVar[str] = "kling-v1"

    async def setup(self, metadata):
        """Initialize configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.metadata = metadata
        self.logger.info(f"Kling {self.MODEL_NAME} Text-to-Video initialized")

    def _build_camera_control(self, input_data: AppInput) -> Optional[dict]:
        """Build camera control configuration."""
        if input_data.camera_type == CameraTypeEnum.none:
            return None
        
        camera_control = {"type": input_data.camera_type.value}
        
        # For "simple" type, build config with camera movements
        if input_data.camera_type == CameraTypeEnum.simple:
            config = {}
            if input_data.camera_horizontal is not None:
                config["horizontal"] = input_data.camera_horizontal
            if input_data.camera_vertical is not None:
                config["vertical"] = input_data.camera_vertical
            if input_data.camera_pan is not None:
                config["pan"] = input_data.camera_pan
            if input_data.camera_tilt is not None:
                config["tilt"] = input_data.camera_tilt
            if input_data.camera_roll is not None:
                config["roll"] = input_data.camera_roll
            if input_data.camera_zoom is not None:
                config["zoom"] = input_data.camera_zoom
            
            if config:
                camera_control["config"] = config
        
        return camera_control

    async def _download_video(self, video_url: str) -> str:
        """Download video and return local path."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name
        
        self.logger.info(f"Downloading video from: {video_url[:100]}...")
        
        async with httpx.AsyncClient(timeout=120.0) as http_client:
            response = await http_client.get(video_url, follow_redirects=True)
            response.raise_for_status()
            
            with open(video_path, "wb") as f:
                f.write(response.content)
        
        self.logger.info(f"Video saved to: {video_path}")
        return video_path

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Kling text-to-video API."""
        try:
            self.logger.info(f"Starting text-to-video generation with prompt: {input_data.prompt[:100]}...")
            
            # Get API credentials
            access_key = os.environ.get("KLING_ACCESS_KEY")
            secret_key = os.environ.get("KLING_SECRET_KEY")
            
            if not access_key or not secret_key:
                raise RuntimeError(
                    "KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables are required."
                )
            
            # Initialize Kling client
            client = KlingClient(access_key=access_key, secret_key=secret_key)
            
            try:
                # Build camera control if specified
                camera_control = self._build_camera_control(input_data)
                
                # Submit task
                self.logger.info("Submitting text-to-video task...")
                response = await client.text_to_video.create_async(
                    prompt=input_data.prompt,
                    duration=input_data.duration.value,
                    model_name=self.MODEL_NAME,
                    negative_prompt=input_data.negative_prompt,
                    mode=input_data.mode.value,
                    aspect_ratio=input_data.aspect_ratio.value,
                    cfg_scale=input_data.cfg_scale,
                    camera_control=camera_control,
                )
                task_id = response.task_id
                self.logger.info(f"Task created with ID: {task_id}")
                
                # Wait for completion using SDK's built-in polling
                self.logger.info("Waiting for task completion...")
                task_result = await client.text_to_video.wait_for_completion_async(
                    task_id=task_id,
                    poll_interval=2.0,
                    timeout=600.0,
                )
                
                # Extract video info
                videos = task_result.task_result.videos
                if not videos:
                    raise RuntimeError("No videos in task result")
                
                video_info = videos[0]
                video_url = video_info.url
                video_id = video_info.id
                duration = getattr(video_info, 'duration', input_data.duration.value)
                
                # Download video
                video_path = await self._download_video(video_url)
                
                self.logger.info("Video generation completed successfully")
                
                return AppOutput(
                    video=File(path=video_path),
                    video_id=video_id,
                    duration=str(duration)
                )
            finally:
                # Clean up client resources
                await client.close_async()

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

