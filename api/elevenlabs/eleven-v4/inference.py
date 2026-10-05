"""Eleven v4: ElevenLabs text to speech on `eleven_v4`.

ElevenLabs' newest and most expressive model: 90+ languages and free-form audio tags.

`run` takes the whole text and returns an audio file. `dialogue` takes the
lines of a conversation, each with its own voice, and returns one file spoken
in a single take. `realtime` is a live function: the caller streams text as it
is written and hears it spoken; what its socket carries is described in
elevenlabs_tts.py.
"""

import logging
from typing import AsyncGenerator, Optional

from inferencesh import BaseApp, Socket
from pydantic import Field

from .elevenlabs_helper import get_api_key
from .elevenlabs_tts import (
    DialogueInput,
    RealtimeInput,
    RealtimeOutput,
    SpeechInput,
    SpeechOutput,
    converse,
    relay,
    speak,
    stop_reading,
)

MODEL = "eleven_v4"
MAX_CHARS = 10000


class AppInput(SpeechInput):
    text: str = Field(description="Text to convert to speech, up to 10,000 characters. Audio tags such as [laughs], [whispers] or [excited] in the text are performed, not read aloud.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = get_api_key()
        self._socket: Optional[Socket] = None

    async def on_cancel(self):
        return await stop_reading(self)

    async def run(self, input_data: AppInput) -> SpeechOutput:
        """Generate speech from text."""
        return await speak(self, MODEL, MAX_CHARS, input_data)

    async def dialogue(self, input_data: DialogueInput) -> SpeechOutput:
        """Generate a conversation between several voices."""
        return await converse(self, MODEL, MAX_CHARS, input_data)

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Speak text as it is written."""
        async for out in relay(self, MODEL, input_data, socket):
            yield out
