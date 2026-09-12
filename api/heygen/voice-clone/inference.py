"""
HeyGen Voice Clone — Instant Clone (Starfish engine).

Clone a voice from a single short recording and, in the same call, speak
arbitrary text in it. Reference audio + text in, speech out.

The cloned voice_id also works anywhere a catalog voice_id does: the
heygen/text-to-speech app and avatar videos.
"""

import logging
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import BaseModel, Field

from .heygen_helper import (
    build_audio_ref,
    delete_endpoint,
    get_client,
    download_file,
    list_voices as fetch_voices,
    poll_voice,
    post_endpoint,
)


class AppInput(BaseAppInput):
    """Clone a voice from a sample, then optionally speak text in it."""

    audio: File = Field(
        description="Reference recording of the voice to clone. 30-60 seconds of clean, "
        "single-speaker speech works best.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Text to speak in the cloned voice. Leave empty to only create the "
        "voice and return its ID.",
        examples=["Merhaba, bu benim klonlanmış sesim. Nasıl duyuluyor?"],
    )
    voice_name: str = Field(
        default="cloned voice",
        max_length=100,
        description="Display name for the cloned voice.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language hint for the clone (e.g. 'tr', 'en'). Auto-detected if omitted.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier for the generated audio (0.5-2.0).",
    )
    remove_background_noise: bool = Field(
        default=True,
        description="Clean background noise out of the reference recording before cloning.",
    )
    keep_voice: bool = Field(
        default=False,
        description="Keep the cloned voice in the workspace so its ID can be reused later. "
        "Off by default: the clone is deleted after the audio is generated, because the "
        "HeyGen account holds a limited number of clone slots. Forced on when no text is given.",
    )


class AppOutput(BaseAppOutput):
    """The cloned voice and, when text was supplied, the speech generated with it."""

    voice_id: str = Field(
        description="Cloned voice ID. Usable with heygen/text-to-speech and avatar videos "
        "while the voice is kept."
    )
    voice_name: str = Field(description="Display name of the cloned voice.")
    kept: bool = Field(
        description="Whether the voice still exists in the workspace. False means the clone "
        "was deleted after generation and the ID is no longer usable."
    )
    audio: Optional[File] = Field(
        default=None, description="Speech generated in the cloned voice (if text was supplied)."
    )
    preview_audio_url: Optional[str] = Field(
        default=None, description="HeyGen's own short preview clip of the cloned voice."
    )


class ListVoicesInput(BaseAppInput):
    """List voices available to this account."""

    voice_type: str = Field(
        default="private",
        description="'private' for this workspace's cloned voices, 'public' for the HeyGen catalog.",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum voices to return.")


class VoiceItem(BaseModel):
    voice_id: str = Field(description="Voice ID — use as voice_id in heygen/text-to-speech")
    name: Optional[str] = Field(default=None, description="Display name")
    language: Optional[str] = Field(default=None, description="Primary language")
    gender: Optional[str] = Field(default=None, description="Voice gender")
    preview_audio_url: Optional[str] = Field(default=None, description="Preview clip URL")


class ListVoicesOutput(BaseAppOutput):
    voices: List[VoiceItem] = Field(description="Matching voices")
    count: int = Field(description="Number of voices returned")


class DeleteVoiceInput(BaseAppInput):
    """Delete a cloned voice to free a clone slot."""

    voice_id: str = Field(description="ID of the cloned voice to delete.")


class DeleteVoiceOutput(BaseAppOutput):
    voice_id: str = Field(description="The deleted voice ID")
    deleted: bool = Field(description="True once the voice no longer exists")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("HeyGen Voice Clone app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        # Without text there is nothing to hand back but the ID, so the voice has to survive.
        keep_voice = input_data.keep_voice or not input_data.text
        if keep_voice and not input_data.keep_voice:
            self.logger.info("No text supplied — keeping the cloned voice so the ID stays usable")

        payload = {
            "audio": build_audio_ref(input_data.audio),
            "voice_name": input_data.voice_name,
            "remove_background_noise": input_data.remove_background_noise,
        }
        if input_data.language:
            payload["language"] = input_data.language

        async with get_client(timeout=300) as client:
            self.logger.info(f"Cloning voice: {input_data.voice_name}")
            created = await post_endpoint(client, "/v3/voices/clone", payload)

            voice_id = created.get("voice_clone_id") or created.get("voice_id")
            if not voice_id:
                raise RuntimeError(f"No voice_clone_id in clone response: {str(created)[:300]}")
            self.logger.info(f"Clone submitted: {voice_id}")

            detail = await poll_voice(client, voice_id)
            self.logger.info(f"Clone ready: {detail.get('name') or input_data.voice_name}")

            audio_file = None
            duration = 0.0
            if input_data.text:
                self.logger.info(f"Generating speech: {input_data.text[:80]}")
                speech = await post_endpoint(
                    client,
                    "/v3/voices/speech",
                    {
                        "text": input_data.text,
                        "voice_id": voice_id,
                        "input_type": "text",
                        "speed": input_data.speed,
                    },
                )
                audio_url = speech.get("audio_url")
                if not audio_url:
                    raise RuntimeError(f"No audio_url in speech response: {str(speech)[:300]}")
                audio_file = File(path=await download_file(audio_url, suffix=".mp3"))
                duration = float(speech.get("duration") or 0.0)
                self.logger.info(f"Speech generated: {duration}s")

            if not keep_voice:
                # Clone slots are a scarce per-account resource shared by every caller.
                self.logger.info(f"Releasing clone slot: deleting voice {voice_id}")
                await delete_endpoint(client, f"/v3/voices/{voice_id}")

        return AppOutput(
            voice_id=voice_id,
            voice_name=detail.get("name") or input_data.voice_name,
            kept=keep_voice,
            audio=audio_file,
            preview_audio_url=detail.get("preview_audio_url"),
            output_meta=OutputMeta(
                inputs=[],
                outputs=[AudioMeta(duration_seconds=duration)] if audio_file else [],
            ),
        )

    async def list_voices(self, input_data: ListVoicesInput) -> ListVoicesOutput:
        async with get_client() as client:
            voices = await fetch_voices(
                client,
                limit=input_data.limit,
                voice_type=input_data.voice_type,
            )

        items = [
            VoiceItem(
                voice_id=v.get("voice_id") or v.get("id") or "",
                name=v.get("name"),
                language=v.get("language"),
                gender=v.get("gender"),
                preview_audio_url=v.get("preview_audio_url"),
            )
            for v in voices
        ]
        self.logger.info(f"Listed {len(items)} {input_data.voice_type} voices")
        return ListVoicesOutput(
            voices=items,
            count=len(items),
            output_meta=OutputMeta(inputs=[], outputs=[]),
        )

    async def delete_voice(self, input_data: DeleteVoiceInput) -> DeleteVoiceOutput:
        async with get_client() as client:
            await delete_endpoint(client, f"/v3/voices/{input_data.voice_id}")
        self.logger.info(f"Deleted voice {input_data.voice_id}")
        return DeleteVoiceOutput(
            voice_id=input_data.voice_id,
            deleted=True,
            output_meta=OutputMeta(inputs=[], outputs=[]),
        )
