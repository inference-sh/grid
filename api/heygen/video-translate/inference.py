"""
HeyGen Video Translation - Translate videos into 30+ languages.

Translates video speech with voice cloning and lip-sync,
preserving the original speaker's voice characteristics.
"""

import logging
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .heygen_helper import (
    get_client,
    post_endpoint,
    poll_translation,
    download_file,
    build_asset_ref,
)


ModeType = Literal["speed", "precision"]


class AppInput(BaseAppInput):
    """Input for video translation."""

    video: File = Field(
        description="Source video to translate. Must be a publicly accessible URL."
    )
    output_language: str = Field(
        description="Target language for translation (e.g., 'Spanish (Spain)', 'Chinese (Mandarin, Simplified)', 'French').",
        examples=["Spanish (Spain)", "Japanese", "German"],
    )
    title: Optional[str] = Field(
        default=None,
        description="Title for the translation job.",
    )
    input_language: Optional[str] = Field(
        default=None,
        description="Source language. Auto-detected if not set.",
    )
    mode: ModeType = Field(
        default="speed",
        description="Translation mode. 'speed' is faster, 'precision' is higher quality.",
    )
    translate_audio_only: bool = Field(
        default=False,
        description="Only translate the audio track, keep original video frames.",
    )
    speaker_num: Optional[int] = Field(
        default=None,
        description="Number of speakers in the video for improved separation.",
    )
    enable_caption: bool = Field(
        default=False,
        description="Generate captions for the translated video.",
    )
    enable_dynamic_duration: bool = Field(
        default=True,
        description="Allow dynamic duration adjustment for natural pacing.",
    )
    disable_music_track: bool = Field(
        default=False,
        description="Remove background music from the output.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start time in seconds for partial translation.",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End time in seconds for partial translation.",
    )


class AppOutput(BaseAppOutput):
    """Output from video translation."""

    video: File = Field(description="The translated video.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Translating video to {input_data.output_language}...")

        video_ref = build_asset_ref(input_data.video)

        payload = {
            "video": video_ref,
            "output_languages": [input_data.output_language],
            "mode": input_data.mode,
            "translate_audio_only": input_data.translate_audio_only,
            "enable_caption": input_data.enable_caption,
            "enable_dynamic_duration": input_data.enable_dynamic_duration,
            "disable_music_track": input_data.disable_music_track,
        }

        if input_data.title:
            payload["title"] = input_data.title
        if input_data.input_language:
            payload["input_language"] = input_data.input_language
        if input_data.speaker_num is not None:
            payload["speaker_num"] = input_data.speaker_num
        if input_data.start_time is not None:
            payload["start_time"] = input_data.start_time
        if input_data.end_time is not None:
            payload["end_time"] = input_data.end_time

        async with get_client(timeout=300) as client:
            result = await post_endpoint(client, "/v3/video-translations", payload)

            translation_ids = result.get("video_translation_ids", [])
            if not translation_ids:
                raise RuntimeError("No translation ID returned from HeyGen API")

            translation_id = translation_ids[0]
            self.logger.info(f"Translation {translation_id} created, polling...")

            completed = await poll_translation(client, translation_id)

        video_url = completed.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL in completed translation")

        video_path = await download_file(video_url)
        duration = completed.get("duration", 0)

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=0,
                    height=0,
                    seconds=float(duration),
                    fps=24,
                    extra={
                        "output_language": input_data.output_language,
                        "mode": input_data.mode,
                    },
                )
            ]
        )

        self.logger.info(f"Translation complete: {duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)
