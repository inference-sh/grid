import logging
import os
from typing import Optional
from enum import Enum

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta, TextMeta
from pydantic import Field

from .minimax_helper import download_file

MUSIC_URL = "https://api.minimax.io/v1/music_generation"
MODEL = "music-3.0"


class FormatEnum(str, Enum):
    mp3 = "mp3"
    wav = "wav"


class AppInput(BaseAppInput):
    """MiniMax Music 3.0 — AI music generation up to 5 minutes.

    Provide a style prompt and/or lyrics. Use section tags in lyrics:
    [Verse], [Chorus], [Bridge], [Intro], [Outro], [Hook], [Pre-Chorus].
    """

    prompt: str = Field(
        description="Style and mood description for the music.",
        examples=["Upbeat pop song with acoustic guitar and light percussion"],
    )
    lyrics: Optional[str] = Field(
        default=None,
        description="Song lyrics with section tags like [Verse], [Chorus]. 1-3500 chars.",
    )
    is_instrumental: bool = Field(
        default=False,
        description="Generate instrumental music without vocals.",
    )
    lyrics_optimizer: bool = Field(
        default=False,
        description="Auto-generate lyrics from the prompt when lyrics field is empty.",
    )
    format: FormatEnum = Field(
        default=FormatEnum.mp3,
        description="Output audio format.",
    )


class AppOutput(BaseAppOutput):
    audio: File = Field(description="The generated music file.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = os.environ.get("MINIMAX_KEY")
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_KEY is not set. Check that `belt secrets get MINIMAX_KEY "
                "--json` reports a non-empty masked_value."
            )
        self.client = httpx.AsyncClient(timeout=300)
        self.logger.info(f"{MODEL} initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Generating music: prompt={input_data.prompt[:80]}, instrumental={input_data.is_instrumental}")

        payload = {
            "model": MODEL,
            "prompt": input_data.prompt,
            "is_instrumental": input_data.is_instrumental,
            "lyrics_optimizer": input_data.lyrics_optimizer,
            "output_format": "url",
            "audio_setting": {
                "format": input_data.format.value,
                "sample_rate": 44100,
                "bitrate": 256000,
            },
        }
        if input_data.lyrics:
            payload["lyrics"] = input_data.lyrics

        resp = await self.client.post(
            MUSIC_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"MiniMax Music error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax Music error: {base_resp.get('status_msg', 'Unknown error')}")

        audio_url = (data.get("data") or {}).get("audio")
        if not audio_url:
            raise RuntimeError(f"No audio URL in response: {str(data)[:300]}")

        extra_info = data.get("extra_info") or {}
        audio_length_ms = float(extra_info.get("audio_length") or 0)
        self.logger.info(f"Music ready: {audio_length_ms}ms")

        ext = input_data.format.value
        audio_path = await download_file(self.client, audio_url, f"/tmp/output.{ext}")

        audio_seconds = audio_length_ms / 1000.0 if audio_length_ms > 0 else 0

        output_meta = OutputMeta(
            inputs=[TextMeta(text=input_data.prompt)],
            outputs=[AudioMeta(
                seconds=audio_seconds,
                sample_rate=44100,
                extra={"model": MODEL},
            )],
        )

        return AppOutput(audio=File(path=audio_path), output_meta=output_meta)

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def unload(self):
        await self.client.aclose()
