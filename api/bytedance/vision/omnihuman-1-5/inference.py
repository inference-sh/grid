"""
OmniHuman 1.5 - BytePlus Audio-Driven Avatar Video Generation

Multi-character audio-driven video generation. Takes a portrait image + audio
and generates a video where the person speaks/sings in sync with the audio.
Supports specifying which character to drive via mask detection.

Uses BytePlus Vision API (different from ARK API).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import ClassVar, Optional
import logging
import os

from .vision_helper import (
    setup_vision_service,
    detect_subjects,
    submit_video_task,
    poll_video_task,
    download_video,
    get_video_duration,
)


class AppInput(BaseAppInput):
    """Input schema for OmniHuman 1.5 avatar video generation."""

    image: File = Field(
        description="Portrait image containing one or more people. JPG format recommended. Max 5MB, max 4096x4096 pixels."
    )
    audio: File = Field(
        description="Audio file to drive the avatar. Duration should be under 15 seconds for best quality."
    )
    mask_index: int = Field(
        default=0,
        ge=0,
        description="Which detected subject to drive (0 = largest face/body, 1 = second largest, etc.). Set to 0 for single-person images."
    )


class AppOutput(BaseAppOutput):
    """Output schema for OmniHuman 1.5 avatar video generation."""

    video: File = Field(description="The generated avatar video file (MP4, 30 FPS).")


class App(BaseApp):
    """OmniHuman 1.5 avatar video generation using BytePlus Vision API."""

    REQ_KEY_DETECT: ClassVar[str] = "realman_avatar_object_detection_cv"
    REQ_KEY_VIDEO: ClassVar[str] = "realman_avatar_picture_omni15_cv"

    async def setup(self):
        """Initialize the BytePlus Vision Service."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.visual_service = setup_vision_service()
        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info("OmniHuman 1.5 initialized")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate avatar video using OmniHuman 1.5."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            self.logger.info("Starting OmniHuman 1.5 avatar video generation")

            # Validate inputs
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            if not input_data.audio.exists():
                raise RuntimeError(f"Input audio does not exist at path: {input_data.audio.path}")

            image_url = input_data.image.uri
            audio_url = input_data.audio.uri

            self.logger.info(f"Image URL: {image_url}")
            self.logger.info(f"Audio URL: {audio_url}")
            self.logger.info(f"Mask index: {input_data.mask_index}")

            # Step 1: Subject detection to get mask URLs
            self.logger.info("Step 1: Detecting subjects in image...")
            mask_urls = detect_subjects(
                self.visual_service,
                image_url,
                logger=self.logger,
            )

            # Select mask URL based on index
            mask_url = None
            if mask_urls:
                if input_data.mask_index >= len(mask_urls):
                    self.logger.warning(
                        f"mask_index {input_data.mask_index} out of range, "
                        f"only {len(mask_urls)} subjects detected. Using index 0."
                    )
                    mask_url = mask_urls[0]
                else:
                    mask_url = mask_urls[input_data.mask_index]
                self.logger.info(f"Using mask URL for subject {input_data.mask_index}")
            else:
                self.logger.info("No subjects detected, proceeding without mask")

            # Check for cancellation
            if self.cancel_flag:
                raise RuntimeError("Task was cancelled by user")

            # Step 2: Submit video generation task
            self.logger.info("Step 2: Submitting video generation task...")
            self.current_task_id = submit_video_task(
                self.visual_service,
                req_key=self.REQ_KEY_VIDEO,
                image_url=image_url,
                audio_url=audio_url,
                mask_url=mask_url,
                logger=self.logger,
            )

            # Step 3: Poll for completion
            self.logger.info("Step 3: Waiting for video generation...")
            result = await poll_video_task(
                self.visual_service,
                req_key=self.REQ_KEY_VIDEO,
                task_id=self.current_task_id,
                logger=self.logger,
                poll_interval=3.0,
                cancel_flag_getter=lambda: self.cancel_flag,
            )

            video_url = result["video_url"]

            # Step 4: Download video
            self.logger.info("Step 4: Downloading generated video...")
            video_path = download_video(video_url, self.logger)

            # Get video duration for billing
            duration = get_video_duration(video_path, self.logger)

            # Build output metadata
            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=0,
                        height=0,
                        seconds=duration,
                        fps=30,
                        extra={
                            "model": "omnihuman-1.5",
                            "mask_index": input_data.mask_index,
                            "subjects_detected": len(mask_urls),
                            "comfyui_cost": result.get("comfyui_cost"),
                            "vid": result.get("vid"),
                        }
                    )
                ]
            )

            self.logger.info(f"Avatar video generated successfully: {video_path}")

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during avatar video generation: {e}")
            raise RuntimeError(f"Avatar video generation failed: {str(e)}")
        finally:
            self.current_task_id = None
