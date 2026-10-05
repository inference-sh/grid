"""Eleven v3: ElevenLabs text to speech on `eleven_v3`.

The previous generation of ElevenLabs' expressive model: 70+ languages and audio tags.

`run` takes the whole text and returns an audio file. `dialogue` takes the
lines of a conversation, each with its own voice, and returns one file spoken
in a single take.
"""

import logging

from inferencesh import BaseApp
from pydantic import Field

from .elevenlabs_helper import get_api_key
from .elevenlabs_tts import DialogueInput, SpeechOutput, StyledSpeechInput, converse, speak

MODEL = "eleven_v3"
MAX_CHARS = 5000


class AppInput(StyledSpeechInput):
    text: str = Field(description="Text to convert to speech, up to 5,000 characters. Audio tags such as [laughs], [whispers] or [excited] in the text are performed, not read aloud.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = get_api_key()

    async def run(self, input_data: AppInput) -> SpeechOutput:
        """Generate speech from text."""
        return await speak(self, MODEL, MAX_CHARS, input_data)

    async def dialogue(self, input_data: DialogueInput) -> SpeechOutput:
        """Generate a conversation between several voices."""
        return await converse(self, MODEL, MAX_CHARS, input_data)
