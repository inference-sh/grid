from inferencesh import BaseApp, BaseAppSetup, File, OutputMeta, VideoMeta
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from .vertex_helper import (
    get_vertex_credentials,
    start_long_running_operation,
    poll_long_running_operation,
    download_video_from_gcs,
    decode_base64_to_bytes,
    save_video_to_temp,
    setup_logger,
    VideoAspectRatioEnum,
    prepare_video_for_veo,
)


class AppSetup(BaseAppSetup):
    """Setup configuration for Veo 2."""
    pass


class RunInput(BaseModel):
    """Input for video generation with Veo 2."""
    prompt: str = Field(
        description="Text prompt describing the desired video content."
    )
    video: Optional[File] = Field(
        None,
        description="Optional video to extend (1-30s MP4, 24fps, 720p/1080p). Extends by 7 seconds."
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Negative prompt to describe what to avoid in the video."
    )
    enhance_prompt: bool = Field(
        default=True,
        description="If true, the prompt will be improved before generating the video."
    )
    aspect_ratio: VideoAspectRatioEnum = Field(
        default=VideoAspectRatioEnum.ratio_16_9,
        description="Video aspect ratio. 16:9 for landscape, 9:16 for portrait."
    )
    duration: int = Field(
        default=8,
        description="Video duration in seconds.",
        ge=5,
        le=8
    )
    num_videos: int = Field(
        default=1,
        description="Number of videos to generate.",
        ge=1,
        le=2
    )


class RunOutput(BaseModel):
    """Output containing generated videos."""
    videos: List[File] = Field(description="The generated video files")


def build_veo2_payload(
    prompt: str,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 8,
    sample_count: int = 1,
    negative_prompt: Optional[str] = None,
    enhance_prompt: bool = True,
    video_path: Optional[str] = None,
    storage_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Build request payload for Veo 2 video generation."""
    instance: Dict[str, Any] = {"prompt": prompt}

    # Add video for extension if provided
    if video_path:
        instance["video"] = prepare_video_for_veo(video_path)

    parameters: Dict[str, Any] = {
        "aspectRatio": aspect_ratio,
        "sampleCount": sample_count,
        "durationSeconds": str(duration_seconds),
        "enhancePrompt": enhance_prompt,
    }

    if negative_prompt:
        parameters["negativePrompt"] = negative_prompt

    if storage_uri:
        parameters["storageUri"] = storage_uri

    return {
        "instances": [instance],
        "parameters": parameters
    }


class App(BaseApp):
    async def setup(self, config: AppSetup):
        """Initialize model configuration."""
        self.logger = setup_logger(__name__)
        self.model_id = "veo-2.0-generate-001"
        self.location = "us-central1"
        self.access_token, self.project = get_vertex_credentials()
        self.logger.info("Veo 2 (Vertex AI) initialized successfully")

    async def run(self, input_data: RunInput) -> RunOutput:
        """Generate video using Veo 2 model via Vertex AI."""
        try:
            self.logger.info(f"Starting video generation with prompt: {input_data.prompt[:100]}...")

            aspect_ratio = input_data.aspect_ratio.value

            # Validate video for extension
            video_path = None
            if input_data.video is not None:
                if not input_data.video.exists():
                    raise RuntimeError(f"Video file does not exist: {input_data.video.path}")
                video_path = input_data.video.path
                self.logger.info(f"Using video for extension: {video_path}")

            self.logger.info(f"Aspect ratio: {aspect_ratio}, Duration: {input_data.duration}s")
            self.logger.info(f"Enhance prompt: {input_data.enhance_prompt}, Num videos: {input_data.num_videos}")
            if input_data.negative_prompt:
                self.logger.info(f"Negative prompt: {input_data.negative_prompt[:50]}...")

            payload = build_veo2_payload(
                prompt=input_data.prompt,
                aspect_ratio=aspect_ratio,
                duration_seconds=input_data.duration,
                sample_count=input_data.num_videos,
                negative_prompt=input_data.negative_prompt,
                enhance_prompt=input_data.enhance_prompt,
                video_path=video_path,
            )

            self.logger.info("Starting video generation operation...")
            operation_response = await start_long_running_operation(
                access_token=self.access_token,
                project=self.project,
                location=self.location,
                model_id=self.model_id,
                payload=payload,
                logger=self.logger
            )

            operation_name = operation_response.get("name")
            if not operation_name:
                raise RuntimeError("No operation name returned from API")

            self.logger.info(f"Polling operation: {operation_name}")
            result = await poll_long_running_operation(
                access_token=self.access_token,
                project=self.project,
                location=self.location,
                model_id=self.model_id,
                operation_name=operation_name,
                poll_interval=5.0,
                max_wait_time=600.0,
                logger=self.logger
            )

            response_data = result.get("response", {})
            videos = response_data.get("videos", [])

            if not videos:
                error = result.get("error")
                if error:
                    raise RuntimeError(f"Video generation failed: {error}")
                raise RuntimeError("No videos in response")

            output_videos = []
            output_meta_videos = []

            for i, video_info in enumerate(videos):
                self.logger.info(f"Processing video {i+1}/{len(videos)}...")

                if "gcsUri" in video_info:
                    video_bytes = await download_video_from_gcs(
                        gcs_uri=video_info["gcsUri"],
                        access_token=self.access_token,
                        logger=self.logger
                    )
                elif "bytesBase64Encoded" in video_info:
                    video_bytes = decode_base64_to_bytes(video_info["bytesBase64Encoded"])
                else:
                    continue

                video_path = save_video_to_temp(video_bytes, "mp4")
                output_videos.append(File(path=video_path))

                # Veo 2 default resolution is approximately 720p
                if aspect_ratio == "16:9":
                    width, height = 1280, 720
                else:
                    width, height = 720, 1280

                output_meta_videos.append(VideoMeta(width=width, height=height, duration=input_data.duration))

            if not output_videos:
                raise RuntimeError("No videos were successfully processed")

            self.logger.info(f"Successfully generated {len(output_videos)} video(s)")

            return RunOutput(
                videos=output_videos,
                output_meta=OutputMeta(outputs=output_meta_videos)
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
