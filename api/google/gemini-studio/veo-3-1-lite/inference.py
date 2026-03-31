from inferencesh import BaseApp, BaseAppSetup, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import BaseModel, Field
from typing import Optional, List

from .gemini_helper import (
    create_gemini_client,
    get_image_dimensions,
    detect_video_aspect_ratio,
    generate_video_with_polling,
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
        self.client = create_gemini_client()
        self.logger.info("Veo 3.1 Lite (Gemini API) initialized successfully")

    async def run(self, input_data: RunInput) -> RunOutput:
        """Generate video using Veo 3.1 Lite model via Gemini API."""
        try:
            self.logger.info(f"Starting video generation with prompt: {input_data.prompt[:100]}...")

            aspect_ratio = input_data.aspect_ratio.value

            image_path = None
            if input_data.image is not None:
                if not input_data.image.exists():
                    raise RuntimeError(f"First frame image does not exist: {input_data.image.path}")
                image_path = input_data.image.path
                img_width, img_height = get_image_dimensions(image_path)
                aspect_ratio = detect_video_aspect_ratio(img_width, img_height)
                self.logger.info(f"Detected aspect ratio from image: {img_width}x{img_height} -> {aspect_ratio}")

            self.logger.info(f"Aspect ratio: {aspect_ratio}, Duration: {input_data.duration}s, Resolution: {input_data.resolution.value}")

            generated_videos = await generate_video_with_polling(
                client=self.client,
                model_id=self.model_id,
                prompt=input_data.prompt,
                image_path=image_path,
                aspect_ratio=aspect_ratio,
                duration_seconds=input_data.duration,
                generate_audio=input_data.generate_audio,
                person_generation=input_data.person_generation.value,
                logger=self.logger,
            )

            output_videos = []
            output_meta_videos = []

            for i, video in enumerate(generated_videos):
                self.logger.info(f"Processing video {i+1}/{len(generated_videos)}...")

                video_bytes = video.video.video_bytes
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

                output_meta_videos.append(VideoMeta(
                    width=width,
                    height=height,
                    seconds=input_data.duration,
                    resolution=input_data.resolution.value,
                    extra={"generate_audio": input_data.generate_audio}
                ))

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
