from inferencesh import BaseApp, BaseAppSetup, BaseAppOutput, File, OutputMeta, VideoMeta
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
    PersonGenerationEnum,
)


class AppSetup(BaseAppSetup):
    """Setup configuration for Veo 3.1 Lite."""
    pass


class RunInput(BaseModel):
    """Input for video generation with Veo 3.1 Lite."""
    prompt: str = Field(
        description="Text prompt describing the desired video content."
    )
    image: Optional[File] = Field(
        None,
        description="Optional first frame image for image-to-video generation."
    )
    aspect_ratio: VideoAspectRatioEnum = Field(
        default=VideoAspectRatioEnum.ratio_16_9,
        description="Video aspect ratio. 16:9 for landscape, 9:16 for portrait."
    )
    duration: int = Field(
        default=8,
        description="Video duration in seconds (4, 6, or 8).",
        ge=4,
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
    person_generation: PersonGenerationEnum = Field(
        default=PersonGenerationEnum.allow_adult,
        description="Person generation setting. allow_adult: only adults, disallow: no people/faces."
    )


class RunOutput(BaseAppOutput):
    """Output containing the generated video."""
    videos: List[File] = Field(description="The generated video files")


class App(BaseApp):
    async def setup(self, config: AppSetup):
        """Initialize model configuration."""
        self.logger = setup_logger(__name__)
        self.model_id = "veo-3.1-lite-generate-preview"
        self.location = "us-central1"
        self.access_token, self.project = get_vertex_credentials()
        self.logger.info("Veo 3.1 Lite (Vertex AI) initialized successfully")

    async def run(self, input_data: RunInput) -> RunOutput:
        """Generate video using Veo 3.1 Lite model via Vertex AI."""
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

            self.logger.info(f"Aspect ratio: {aspect_ratio}, Duration: {input_data.duration}s, Resolution: {input_data.resolution.value}")

            payload = build_veo_payload(
                prompt=input_data.prompt,
                aspect_ratio=aspect_ratio,
                duration_seconds=input_data.duration,
                resolution=input_data.resolution.value,
                generate_audio=input_data.generate_audio,
                sample_count=1,
                first_frame_path=first_frame_path,
                person_generation=input_data.person_generation.value,
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

            self.logger.info(f"Full operation result keys: {result.keys()}")
            self.logger.info(f"Response data keys: {response_data.keys() if response_data else 'empty'}")
            self.logger.info(f"Number of videos in response: {len(videos)}")

            if not videos:
                self.logger.error(f"Full result: {result}")
                error = result.get("error")
                if error:
                    raise RuntimeError(f"Video generation failed: {error}")
                rai_reasons = response_data.get("raiMediaFilteredReasons", [])
                rai_count = response_data.get("raiMediaFilteredCount", 0)
                if rai_reasons:
                    raise RuntimeError(f"Video was blocked by content filtering ({rai_count} filtered): {'; '.join(rai_reasons)}")
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
                    if input_data.resolution.value == "4k":
                        width, height = 3840, 2160
                    elif input_data.resolution.value == "1080p":
                        width, height = 1920, 1080
                    else:
                        width, height = 1280, 720
                else:
                    if input_data.resolution.value == "4k":
                        width, height = 2160, 3840
                    elif input_data.resolution.value == "1080p":
                        width, height = 1080, 1920
                    else:
                        width, height = 720, 1280

                output_meta_videos.append(VideoMeta(width=width, height=height, seconds=input_data.duration, resolution=input_data.resolution.value, extra={"generate_audio": input_data.generate_audio}))

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
