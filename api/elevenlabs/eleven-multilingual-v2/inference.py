"""Eleven Multilingual v2: ElevenLabs text to speech on `eleven_multilingual_v2`.

ElevenLabs' stable, high-quality multilingual model for long-form speech.
"""

import logging

from inferencesh import BaseApp
from pydantic import Field

from .elevenlabs_helper import get_api_key
from .elevenlabs_tts import SpeechOutput, StyledSpeechInput, speak

MODEL = "eleven_multilingual_v2"
MAX_CHARS = 10000


class AppInput(StyledSpeechInput):
    text: str = Field(description="Text to convert to speech, up to 10,000 characters.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = get_api_key()

    async def run(self, input_data: AppInput) -> SpeechOutput:
        """Generate speech from text."""
        return await speak(self, MODEL, MAX_CHARS, input_data)
