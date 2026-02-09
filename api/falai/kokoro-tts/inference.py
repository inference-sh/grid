"""
Kokoro TTS - Lightweight Text-to-Speech

Fast, high-quality text-to-speech supporting 9 languages with
multiple voice options per language.

Consolidates fal-ai/kokoro/* endpoints into a single app.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, AudioMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_file

# Suppress noisy httpx polling logs
logging.getLogger("httpx").setLevel(logging.WARNING)


# Language to fal.ai endpoint mapping
LANGUAGE_ENDPOINTS = {
    "american-english": "fal-ai/kokoro/american-english",
    "british-english": "fal-ai/kokoro/british-english",
    "french": "fal-ai/kokoro/french",
    "spanish": "fal-ai/kokoro/spanish",
    "japanese": "fal-ai/kokoro/japanese",
    "italian": "fal-ai/kokoro/italian",
    "hindi": "fal-ai/kokoro/hindi",
    "brazilian-portuguese": "fal-ai/kokoro/brazilian-portuguese",
    "mandarin-chinese": "fal-ai/kokoro/mandarin-chinese",
}

# Available voices per language
LANGUAGE_VOICES = {
    "american-english": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
    ],
    "british-english": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "french": ["ff_siwis"],
    "spanish": ["ef_dora", "em_alex", "em_santa"],
    "japanese": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "italian": ["if_sara", "im_nicola"],
    "hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "brazilian-portuguese": ["pf_dora", "pm_alex", "pm_santa"],
    "mandarin-chinese": [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
}

# Default voice per language
DEFAULT_VOICES = {
    "american-english": "af_heart",
    "british-english": "bf_alice",
    "french": "ff_siwis",
    "spanish": "ef_dora",
    "japanese": "jf_alpha",
    "italian": "if_sara",
    "hindi": "hf_alpha",
    "brazilian-portuguese": "pf_dora",
    "mandarin-chinese": "zf_xiaobei",
}


class AppInput(BaseAppInput):
    """Input schema for Kokoro TTS."""
    prompt: str = Field(
        description="Text to convert to speech."
    )
    language: Literal[
        "american-english",
        "british-english",
        "french",
        "spanish",
        "japanese",
        "italian",
        "hindi",
        "brazilian-portuguese",
        "mandarin-chinese",
    ] = Field(
        default="american-english",
        description="Language for speech generation."
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice ID. If not set, uses the default voice for the selected language. See docs for available voices per language."
    )
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Speed of the generated audio. Default is 1.0."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kokoro TTS."""
    audio: File = Field(description="The generated speech audio")


class App(BaseApp):
    """Kokoro TTS app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kokoro TTS app initialized")

    def _get_model_id(self, input_data: AppInput) -> str:
        """Select fal.ai endpoint based on language."""
        return LANGUAGE_ENDPOINTS[input_data.language]

    def _resolve_voice(self, input_data: AppInput) -> str:
        """Resolve voice ID, validating against available voices."""
        if input_data.voice:
            valid_voices = LANGUAGE_VOICES[input_data.language]
            if input_data.voice not in valid_voices:
                raise ValueError(
                    f"Voice '{input_data.voice}' is not available for {input_data.language}. "
                    f"Available voices: {', '.join(valid_voices)}"
                )
            return input_data.voice
        return DEFAULT_VOICES[input_data.language]

    def _build_request(self, input_data: AppInput) -> dict:
        """Build request payload for fal.ai."""
        return {
            "prompt": input_data.prompt,
            "voice": self._resolve_voice(input_data),
            "speed": input_data.speed,
        }

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run text-to-speech inference."""
        try:
            setup_fal_client()

            model_id = self._get_model_id(input_data)
            voice = self._resolve_voice(input_data)
            self.logger.info(f"Language: {input_data.language}, Voice: {voice}, Model: {model_id}")
            self.logger.info(f"Processing text: {input_data.prompt[:100]}...")

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            # Download the generated audio
            audio_url = result["audio"]["url"]
            audio_path = download_file(audio_url, suffix=".wav", logger=self.logger)

            return AppOutput(
                audio=File(path=audio_path),
                output_meta=OutputMeta(
                    inputs=[TextMeta(text=input_data.prompt)],
                    outputs=[AudioMeta(seconds=0.0)]
                )
            )

        except ValueError as e:
            raise
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"TTS generation failed: {str(e)}")
