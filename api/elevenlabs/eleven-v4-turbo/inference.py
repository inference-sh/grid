"""Eleven v4 Turbo: ElevenLabs text to speech on `eleven_v4_turbo`.

The low-latency v4 model, for speech that has to start right away.

`run` takes the whole text and returns an audio file. `realtime` is a live
function: the caller streams text as it is written and hears it spoken; what
its socket carries is described in elevenlabs_tts.py.
"""

import logging
from typing import AsyncGenerator, Optional

from inferencesh import BaseApp, Socket
from pydantic import Field

from .elevenlabs_helper import get_api_key
from .elevenlabs_tts import RealtimeInput, RealtimeOutput, SpeechInput, SpeechOutput, relay, speak, stop_reading

MODEL = "eleven_v4_turbo"
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

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Speak text as it is written."""
        async for out in relay(self, MODEL, input_data, socket):
            yield out
