"""
Kling Video-to-Audio

Add sound effects and background music to videos using Kling's audio generation
model via fal.ai. Returns both the dubbed video and extracted audio.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video, download_file

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ENDPOINT = "fal-ai/kling-video/video-to-audio"


class AppInput(BaseAppInput):
    """Input schema for Kling Video-to-Audio."""

    video: File = Field(
        description="Input video to add audio to. Supported formats: MP4, MOV.",
    )
    sound_effect_prompt: str = Field(
        default="Car tires screech, engine revving loudly, gravel crunching under wheels",
        description="Prompt describing the sound effects to generate for the video.",
    )
    background_music_prompt: str = Field(
        default="intense car race",
        description="Prompt describing the background music style.",
    )
    asmr_mode: bool = Field(
        default=False,
        description="Enable ASMR mode for softer, more intimate audio generation.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Video-to-Audio."""

    video: File = Field(description="The video with generated audio dubbed in.")
    audio: File = Field(description="The extracted audio track in MP3 format.")


class App(BaseApp):
    """Kling Video-to-Audio application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Video-to-Audio initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        return {
            "video_url": input_data.video.uri,
            "sound_effect_prompt": input_data.sound_effect_prompt,
            "background_music_prompt": input_data.background_music_prompt,
            "asmr_mode": input_data.asmr_mode,
        }

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

            setup_fal_client()

            self.logger.info(f"Adding audio to video: sfx={input_data.sound_effect_prompt[:60]}...")
            self.logger.info(f"Settings: music={input_data.background_music_prompt}, asmr={input_data.asmr_mode}")

            request_data = self._build_request(input_data)
            result = run_fal_model(MODEL_ENDPOINT, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            audio_url = result["audio"]["url"]
            audio_path = download_file(audio_url, suffix=".mp3", logger=self.logger)

            output_meta = OutputMeta(
                outputs=[AudioMeta(seconds=0.0)]
            )

            return AppOutput(
                video=File(path=video_path),
                audio=File(path=audio_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during audio generation: {e}")
            raise RuntimeError(f"Audio generation failed: {str(e)}")
