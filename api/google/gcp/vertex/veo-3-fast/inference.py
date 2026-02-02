from inferencesh import BaseApp, BaseAppSetup, File, OutputMeta, VideoMeta
from pydantic import BaseModel, Field
from typing import Optional, List

from .vertex_helper import (
    get_vertex_credentials,
    get_image_dimensions,
    detect_video_aspect_ratio,
    build_veo_payload,
    start_long_running_operation,
    poll_long_running_operation,
    download_video_from_gcs,
    decode_base64_to_bytes,
    save_video_to_temp,
    setup_logger,
    VideoAspectRatioEnum,
    VideoResolutionEnum,
)


class AppSetup(BaseAppSetup):
    """Setup configuration for Veo 3 Fast."""
    pass


class RunInput(BaseModel):
    """Input for video generation with Veo 3 Fast."""
    prompt: str = Field(
        description="Text prompt describing the desired video content."
    )
    image: Optional[File] = Field(
        None,
        description="Optional first frame image for image-to-video generation."
    )
    last_frame: Optional[File] = Field(
        None,
        description="Optional last frame image for frame interpolation. Requires first frame image."
    )
    video: Optional[File] = Field(
        None,
        description="Optional video to extend (1-30s MP4, 24fps, 720p/1080p). Extends by 7 seconds."
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
    resolution: VideoResolutionEnum = Field(
        default=VideoResolutionEnum.res_720p,
        description="Output video resolution."
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether to generate audio for the video."
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


class App(BaseApp):
    async def setup(self, config: AppSetup):
        """Initialize model configuration."""
        self.logger = setup_logger(__name__)
        self.model_id = "veo-3.0-fast-generate-001"
        self.location = "us-central1"
        self.access_token, self.project = get_vertex_credentials()
        self.logger.info("Veo 3 Fast (Vertex AI) initialized successfully")

    async def run(self, input_data: RunInput) -> RunOutput:
        """Generate video using Veo 3 Fast model via Vertex AI."""
        try:
            self.logger.info(f"Starting video generation with prompt: {input_data.prompt[:100]}...")

            aspect_ratio = input_data.aspect_ratio.value

            first_frame_path = None
            if input_data.image is not None:
                if not input_data.image.exists():
                    raise RuntimeError(f"First frame image does not exist: {input_data.image.path}")
                first_frame_path = input_data.image.path
                img_width, img_height = get_image_dimensions(first_frame_path)
                aspect_ratio = detect_video_aspect_ratio(img_width, img_height)
                self.logger.info(f"Detected aspect ratio from image: {img_width}x{img_height} -> {aspect_ratio}")

            last_frame_path = None
            if input_data.last_frame is not None:
                if first_frame_path is None:
                    raise RuntimeError("Last frame requires first frame image to be provided")
                if not input_data.last_frame.exists():
                    raise RuntimeError(f"Last frame image does not exist: {input_data.last_frame.path}")
                last_frame_path = input_data.last_frame.path
                self.logger.info("Using last frame for frame interpolation")

            video_path = None
            if input_data.video is not None:
                if not input_data.video.exists():
                    raise RuntimeError(f"Video file does not exist: {input_data.video.path}")
                if first_frame_path is not None or last_frame_path is not None:
                    raise RuntimeError("Video extension cannot be combined with first/last frame images")
                video_path = input_data.video.path
                self.logger.info(f"Using video for extension: {video_path}")

            self.logger.info(f"Aspect ratio: {aspect_ratio}, Duration: {input_data.duration}s, Resolution: {input_data.resolution.value}")

            payload = build_veo_payload(
                prompt=input_data.prompt,
                aspect_ratio=aspect_ratio,
                duration_seconds=input_data.duration,
                resolution=input_data.resolution.value,
                generate_audio=input_data.generate_audio,
                sample_count=input_data.num_videos,
                first_frame_path=first_frame_path,
                last_frame_path=last_frame_path,
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

                if aspect_ratio == "16:9":
                    width, height = (1920, 1080) if input_data.resolution.value == "1080p" else (1280, 720)
                else:
                    width, height = (1080, 1920) if input_data.resolution.value == "1080p" else (720, 1280)

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
