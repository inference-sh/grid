"""
Shared helper module for Inworld AI API operations.
Uses httpx for direct REST API access with Basic auth.

Symlink this file into your app folder for deployment.
"""

import os
import base64
import logging
import tempfile
import subprocess
import json
from typing import Optional, Dict, Any

import httpx


TTS_BASE_URL = "https://api.inworld.ai/tts/v1"
STT_BASE_URL = "https://api.inworld.ai/stt/v1"


def get_api_key() -> str:
    """Validate API key exists."""
    key = os.environ.get("INWORLD_KEY")
    if not key:
        raise RuntimeError("INWORLD_KEY environment variable is required")
    return key


def get_auth_header() -> str:
    """Get Basic auth header value. Inworld portal provides a pre-encoded Base64 key."""
    key = get_api_key()
    return f"Basic {key}"


def get_audio_duration(file_path: str, logger: Optional[logging.Logger] = None) -> float:
    """Get duration of an audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", file_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0))
            if logger:
                logger.info(f"Audio duration: {duration:.2f}s")
            return duration
    except Exception as e:
        if logger:
            logger.warning(f"Could not get audio duration: {e}")
    return 0.0


async def text_to_speech(
    text: str,
    voice_id: str,
    model_id: str,
    audio_encoding: str = "MP3",
    sample_rate_hertz: int = 44100,
    speaking_rate: float = 1.0,
    delivery_mode: Optional[str] = None,
    temperature: Optional[float] = None,
    language: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Convert text to speech using Inworld TTS API.

    Returns path to the generated audio file.
    """
    if logger:
        logger.info(f"Generating speech with model: {model_id}, voice: {voice_id}")

    body: Dict[str, Any] = {
        "text": text,
        "voiceId": voice_id,
        "modelId": model_id,
        "audioConfig": {
            "audioEncoding": audio_encoding,
            "sampleRateHertz": sample_rate_hertz,
            "speakingRate": speaking_rate,
        },
    }

    if delivery_mode and delivery_mode != "DELIVERY_MODE_UNSPECIFIED":
        body["deliveryMode"] = delivery_mode
    if temperature is not None:
        body["temperature"] = temperature
    if language:
        body["language"] = language

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{TTS_BASE_URL}/voice",
            headers={
                "Authorization": get_auth_header(),
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

    data = response.json()
    audio_b64 = data.get("audioContent", "")
    if not audio_b64:
        raise RuntimeError("No audioContent in response")

    audio_bytes = base64.b64decode(audio_b64)

    ext_map = {
        "MP3": ".mp3",
        "WAV": ".wav",
        "OGG_OPUS": ".ogg",
        "FLAC": ".flac",
        "LINEAR16": ".pcm",
        "PCM": ".pcm",
        "ALAW": ".alaw",
        "MULAW": ".mulaw",
    }
    suffix = ext_map.get(audio_encoding, ".mp3")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        file_path = tmp.name

    if logger:
        logger.info(f"Audio saved to: {file_path} ({len(audio_bytes)} bytes)")

    return file_path


async def speech_to_text(
    audio_path: str,
    model_id: str = "inworld/inworld-stt-1",
    audio_encoding: str = "AUTO_DETECT",
    language: Optional[str] = None,
    sample_rate_hertz: int = 16000,
    include_word_timestamps: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Transcribe audio using Inworld STT API.

    Returns the full API response dict.
    """
    if logger:
        logger.info(f"Transcribing audio: {audio_path} with model: {model_id}")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode()

    body: Dict[str, Any] = {
        "transcribeConfig": {
            "modelId": model_id,
            "audioEncoding": audio_encoding,
            "sampleRateHertz": sample_rate_hertz,
            "includeWordTimestamps": include_word_timestamps,
        },
        "audioData": {
            "content": audio_b64,
        },
    }

    if language:
        body["transcribeConfig"]["language"] = language

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{STT_BASE_URL}/transcribe",
            headers={
                "Authorization": get_auth_header(),
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

    result = response.json()

    if logger:
        transcript = result.get("transcription", {}).get("transcript", "")
        logger.info(f"Transcription complete: {len(transcript)} characters")

    return result
