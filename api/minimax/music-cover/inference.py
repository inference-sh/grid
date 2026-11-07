import logging
import os
from typing import Optional
from enum import Enum

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field

from .minimax_helper import download_file

MUSIC_URL = "https://api.minimax.io/v1/music_generation"
PREPROCESS_URL = "https://api.minimax.io/v1/music_cover_preprocess"
MODEL = "music-cover"


class FormatEnum(str, Enum):
    mp3 = "mp3"
    wav = "wav"


class AppInput(BaseAppInput):
    """MiniMax Music Cover — AI-powered song covers and style transfer.

    Upload a reference track to extract its style, then generate a cover
    with new lyrics and/or style direction.
    """

    reference_audio: File = Field(
        description="Reference audio track for style transfer. 6s-6min, max 50MB.",
    )
    prompt: str = Field(
        description="Style and mood direction for the cover.",
        examples=["Jazz arrangement with piano and soft drums"],
    )
    lyrics: Optional[str] = Field(
        default=None,
        description="New lyrics for the cover with section tags. 10-1000 chars. If empty, uses reference lyrics.",
    )
    format: FormatEnum = Field(
        default=FormatEnum.mp3,
        description="Output audio format.",
    )


class AppOutput(BaseAppOutput):
    audio: File = Field(description="The generated cover audio file.")


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
        self.logger.info(f"Preprocessing reference audio...")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        preprocess_resp = await self.client.post(
            PREPROCESS_URL,
            headers=headers,
            json={
                "model": MODEL,
                "audio_url": input_data.reference_audio.uri,
            },
        )

        if preprocess_resp.status_code != 200:
            raise RuntimeError(f"Preprocess error ({preprocess_resp.status_code}): {preprocess_resp.text[:300]}")

        preprocess_data = preprocess_resp.json()
        # MiniMax signals success with base_resp.status_code == 0. A missing or
        # non-zero status means the call did not succeed (e.g. an account without
        # music API access), so surface the whole body instead of masking it.
        base_resp = preprocess_data.get("base_resp") or {}
        if base_resp.get("status_code") != 0:
            detail = base_resp.get("status_msg") or "unknown error"
            raise RuntimeError(f"MiniMax preprocess failed: {detail} | body: {str(preprocess_data)[:500]}")

        # cover_feature_id, formatted_lyrics and audio_duration are returned at the
        # top level of the response, not inside a "data" object.
        cover_feature_id = preprocess_data.get("cover_feature_id")
        if not cover_feature_id:
            raise RuntimeError(f"MiniMax preprocess returned no cover_feature_id | body: {str(preprocess_data)[:500]}")

        formatted_lyrics = preprocess_data.get("formatted_lyrics", "")
        audio_duration = float(preprocess_data.get("audio_duration") or 0)
        self.logger.info(f"Preprocessed: feature_id={cover_feature_id[:20]}..., ref_duration={audio_duration}s")

        lyrics = input_data.lyrics or formatted_lyrics
        if not lyrics:
            lyrics = "[Verse]\nLa la la"

        self.logger.info(f"Generating cover: prompt={input_data.prompt[:80]}")

        payload = {
            "model": MODEL,
            "prompt": input_data.prompt,
            "lyrics": lyrics,
            "cover_feature_id": cover_feature_id,
            "output_format": "url",
            "audio_setting": {
                "format": input_data.format.value,
                "sample_rate": 44100,
                "bitrate": 256000,
            },
        }

        resp = await self.client.post(MUSIC_URL, headers=headers, json=payload)

        if resp.status_code != 200:
            raise RuntimeError(f"Music cover error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code") != 0:
            detail = base_resp.get("status_msg") or "unknown error"
            raise RuntimeError(f"MiniMax music cover failed: {detail} | body: {str(data)[:500]}")

        audio_url = (data.get("data") or {}).get("audio")
        if not audio_url:
            raise RuntimeError(f"MiniMax music cover returned no audio URL | body: {str(data)[:500]}")

        extra_info = data.get("extra_info") or {}
        # MiniMax reports duration as extra_info.music_duration (milliseconds).
        audio_length_ms = float(extra_info.get("music_duration") or 0)
        self.logger.info(f"Cover ready: {audio_length_ms}ms")

        ext = input_data.format.value
        audio_path = await download_file(self.client, audio_url, f"/tmp/output.{ext}")

        audio_seconds = audio_length_ms / 1000.0 if audio_length_ms > 0 else 0

        output_meta = OutputMeta(
            inputs=[AudioMeta(seconds=audio_duration)],
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
