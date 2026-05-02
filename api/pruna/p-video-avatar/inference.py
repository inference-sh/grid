"""
P-Video-Avatar - Talking head video generation from a portrait image by Pruna
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_video, upload_file


class VoiceEnum(str, Enum):
    zephyr = "Zephyr (Female)"
    puck = "Puck (Male)"
    charon = "Charon (Male)"
    kore = "Kore (Female)"
    fenrir = "Fenrir (Male)"
    leda = "Leda (Female)"
    orus = "Orus (Male)"
    aoede = "Aoede (Female)"
    callirrhoe = "Callirrhoe (Female)"
    autonoe = "Autonoe (Female)"
    enceladus = "Enceladus (Male)"
    iapetus = "Iapetus (Male)"
    umbriel = "Umbriel (Male)"
    algenib = "Algenib (Male)"
    despina = "Despina (Female)"
    erinome = "Erinome (Female)"
    laomedeia = "Laomedeia (Female)"
    achernar = "Achernar (Female)"
    algieba = "Algieba (Male)"
    schedar = "Schedar (Male)"
    gacrux = "Gacrux (Female)"
    pulcherrima = "Pulcherrima (Female)"
    achird = "Achird (Male)"
    zubenelgenubi = "Zubenelgenubi (Male)"
    vindemiatrix = "Vindemiatrix (Female)"
    sadachbia = "Sadachbia (Male)"
    sadaltager = "Sadaltager (Male)"
    sulafat = "Sulafat (Female)"
    alnilam = "Alnilam (Male)"
    rasalgethi = "Rasalgethi (Male)"


class VoiceLanguageEnum(str, Enum):
    en_us = "English (US)"
    en_uk = "English (UK)"
    spanish = "Spanish"
    french = "French"
    german = "German"
    italian = "Italian"
    portuguese_br = "Portuguese (Brazil)"
    japanese = "Japanese"
    korean = "Korean"
    hindi = "Hindi"


class ResolutionEnum(str, Enum):
    hd = "720p"
    full_hd = "1080p"


class AppInput(BaseAppInput):
    """Input schema for P-Video-Avatar."""

    image: File = Field(
        description="Portrait image (first frame). Supports jpg, jpeg, png, webp."
    )
    voice_script: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Script for the person to say. Required if no audio provided."
    )
    audio: Optional[File] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Audio file to drive speech. If both audio and voice_script are provided, audio is used."
    )

    model_config = {
        "json_schema_extra": {
            "anyOf": [
                {"properties": {"voice_script": {"not": {"type": "null"}}}},
                {"properties": {"audio": {"not": {"type": "null"}}}}
            ]
        }
    }
    voice: VoiceEnum = Field(
        default=VoiceEnum.zephyr,
        description="Voice for generated speech (used with voice_script)."
    )
    voice_language: VoiceLanguageEnum = Field(
        default=VoiceLanguageEnum.en_us,
        description="Output language for generated speech."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.hd,
        description="Video resolution: 720p or 1080p."
    )
    video_prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt for the video (default: 'The person is talking.')."
    )
    voice_prompt: Optional[str] = Field(
        default=None,
        description="Optional speaking style, tone, pacing or emotion instructions."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    disable_safety_filter: bool = Field(
        default=True,
        description="Disable safety filter for prompts and input image."
    )
    disable_prompt_upsampling: bool = Field(
        default=False,
        description="Skip the multimodal prompt upsampler and use raw prompt."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Video-Avatar."""

    video: File = Field(description="Generated talking head video.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Video-Avatar for talking head video generation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-avatar"
        self.logger.info("P-Video-Avatar initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate talking head video."""
        if not input_data.voice_script and not input_data.audio:
            raise RuntimeError("Either voice_script or audio must be provided")

        self.logger.info(f"Generating avatar video, resolution: {input_data.resolution.value}")

        # Build request
        request_data = {
            "voice": input_data.voice.value,
            "voice_language": input_data.voice_language.value,
            "resolution": input_data.resolution.value,
            "disable_safety_filter": input_data.disable_safety_filter,
            "disable_prompt_upsampling": input_data.disable_prompt_upsampling,
        }

        # Handle image input (required)
        if not input_data.image.exists():
            raise RuntimeError(f"Input image does not exist: {input_data.image.path}")

        if input_data.image.uri and input_data.image.uri.startswith("http"):
            request_data["image"] = input_data.image.uri
        else:
            self.logger.info("Uploading portrait image...")
            upload_result = upload_file(input_data.image.path, logger=self.logger)
            image_url = upload_result.get("urls", {}).get("get")
            if not image_url:
                raise RuntimeError("Failed to get URL for uploaded image")
            request_data["image"] = image_url

        # Handle audio input
        if input_data.audio:
            if not input_data.audio.exists():
                raise RuntimeError(f"Audio file does not exist: {input_data.audio.path}")

            if input_data.audio.uri and input_data.audio.uri.startswith("http"):
                request_data["audio"] = input_data.audio.uri
            else:
                self.logger.info("Uploading audio file...")
                upload_result = upload_file(input_data.audio.path, logger=self.logger)
                audio_url = upload_result.get("urls", {}).get("get")
                if not audio_url:
                    raise RuntimeError("Failed to get URL for uploaded audio")
                request_data["audio"] = audio_url

        # Handle voice_script
        if input_data.voice_script:
            request_data["voice_script"] = input_data.voice_script
            self.logger.info(f"Voice script: {input_data.voice_script[:80]}...")

        if input_data.video_prompt:
            request_data["video_prompt"] = input_data.video_prompt

        if input_data.voice_prompt:
            request_data["voice_prompt"] = input_data.voice_prompt

        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        # Avatar generation is slower - use async polling
        result = await run_prediction(
            model=self.model,
            input_data=request_data,
            use_sync=False,
            logger=self.logger,
        )

        # Download result
        generation_url = get_generation_url(result)
        video_path = download_video(generation_url, logger=self.logger)

        # Build output metadata for pricing
        # Avatar videos are always portrait-ish (talking head)
        resolution_map = {
            "720p": VideoResolution.VIDEO_RES720_P,
            "1080p": VideoResolution.VIDEO_RES1080_P,
        }

        # Default dimensions (portrait orientation for avatar)
        dims_map = {
            "720p": (720, 1280),
            "1080p": (1080, 1920),
        }
        width, height = dims_map.get(input_data.resolution.value, (720, 1280))

        # Try to get actual video duration from result
        video_seconds = float(result.get("duration", 10))

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width,
                    height=height,
                    resolution=resolution_map.get(input_data.resolution.value, VideoResolution.VIDEO_RES720_P),
                    seconds=video_seconds,
                )
            ],
        )

        self.logger.info(f"Avatar video generated: {video_seconds}s at {input_data.resolution.value}")

        return AppOutput(
            video=File(path=video_path),
            seed=result.get("seed"),
            output_meta=output_meta,
        )
