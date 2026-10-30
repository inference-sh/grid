"""
P-Video-2-Pro-Avatar - Talking video from a portrait and audio or a script by Pruna (MiniMax H3)
"""

import logging
from enum import Enum
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta, VideoMeta
from pydantic import Field, model_validator

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video, probe_media_duration


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
    r480 = "480p"
    r768 = "768p"
    r1080 = "1080p"
    r2k = "2k"
    r4k = "4k"


class AppInput(BaseAppInput):
    image: File = Field(description="Portrait of the person who speaks (jpg, png, webp; up to 20 MiB). Sets the aspect ratio.")
    audio: Optional[File] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Speech audio, 1s to 3 min (64 MiB). The video matches its length. Overrides voice_script.",
    )
    voice_script: Optional[str] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Exact words to speak with text-to-speech when no audio is uploaded.",
    )
    voice: VoiceEnum = Field(default=VoiceEnum.zephyr, description="TTS voice for voice_script.")
    voice_language: VoiceLanguageEnum = Field(default=VoiceLanguageEnum.en_us, description="Language of voice_script.")
    voice_prompt: Optional[str] = Field(
        default=None,
        description="Delivery instructions for the voice (tone, pacing, accent, emotion). Not spoken.",
    )
    prompt: Optional[str] = Field(default=None, description="What the video should show (default: 'The person is talking.').")
    prompt_upsampling: bool = Field(
        default=True,
        description="Append the scene's own motion from the portrait to the prompt. Your words stay first.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r768,
        description="480p or 768p native. 1080p, 2k and 4k generate at 768p and upscale; 4k up to 125s.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    enable_safety_filter: bool = Field(
        default=False,
        description="Block content in the toggleable moderation categories. Always-on rules apply either way.",
    )

    model_config = {
        "json_schema_extra": {
            "anyOf": [
                {"properties": {"voice_script": {"not": {"type": "null"}}}},
                {"properties": {"audio": {"not": {"type": "null"}}}},
            ]
        }
    }

    @model_validator(mode="after")
    def check_speech(self):
        if not self.audio and not (self.voice_script and self.voice_script.strip()):
            raise ValueError("Provide audio or voice_script")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated talking video.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-2-pro-avatar"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "image": file_url(input_data.image, logger=self.logger),
            "voice": input_data.voice.value,
            "voice_language": input_data.voice_language.value,
            "resolution": input_data.resolution.value,
            "prompt_upsampling": input_data.prompt_upsampling,
            "enable_safety_filter": input_data.enable_safety_filter,
        }
        if input_data.audio:
            request_data["audio"] = file_url(input_data.audio, logger=self.logger)
        elif input_data.voice_script:
            request_data["voice_script"] = input_data.voice_script
        if input_data.voice_prompt:
            request_data["voice_prompt"] = input_data.voice_prompt
        if input_data.prompt:
            request_data["prompt"] = input_data.prompt
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        self.logger.info(
            "p-video-2-pro-avatar: %s, audio=%s, script_chars=%d",
            input_data.resolution.value, bool(input_data.audio), len(input_data.voice_script or ""),
        )

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        # The soundtrack sets the length; bill the finished video's seconds
        info = probe_video(video_path, logger=self.logger)
        seconds = info["seconds"]
        if seconds <= 0 and input_data.audio:
            seconds = probe_media_duration(input_data.audio.path, logger=self.logger)
        if seconds <= 0:
            self.logger.warning("Could not determine output video duration, billing 1s minimum")
            seconds = 1.0

        inputs = []
        if input_data.audio:
            inputs.append(AudioMeta(seconds=probe_media_duration(input_data.audio.path, logger=self.logger)))

        output_meta = OutputMeta(
            inputs=inputs,
            outputs=[
                VideoMeta(
                    width=info["width"],
                    height=info["height"],
                    resolution=input_data.resolution.value,
                    seconds=round(seconds, 3),
                    fps=24,
                    extra={"resolution": input_data.resolution.value},
                )
            ],
        )
        self.logger.info("Avatar video: %sx%s, %.2fs", info["width"], info["height"], seconds)
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)
