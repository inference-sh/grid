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
VOICES_BASE_URL = "https://api.inworld.ai/voices/v1"


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


async def list_voices(
    language: Optional[str] = None,
    gender: Optional[str] = None,
    source: Optional[str] = None,
    page_size: int = 2000,
    logger: Optional[logging.Logger] = None,
) -> list:
    """List available voices from Inworld Voices API.

    Returns list of voice dicts with voiceId, displayName, langCode, source, etc.
    """
    if logger:
        logger.info("Listing available voices")

    params: Dict[str, Any] = {"pageSize": page_size}

    filters = []
    if language:
        filters.append(f'langCode="{language}"')
    if gender:
        filters.append(f'gender="{gender}"')
    if source:
        filters.append(f'source="{source}"')
    if filters:
        params["filter"] = " AND ".join(filters)

    all_voices = []
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            response = await client.get(
                f"{VOICES_BASE_URL}/voices",
                headers={
                    "Authorization": get_auth_header(),
                    "Content-Type": "application/json",
                },
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            all_voices.extend(data.get("voices", []))
            next_token = data.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token

    if logger:
        logger.info(f"Found {len(all_voices)} voices")

    return all_voices


async def design_voice(
    design_prompt: str,
    preview_text: str,
    lang_code: str = "EN_US",
    number_of_samples: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Design a voice from a text description using Inworld Voice Design API.

    Returns up to 3 preview voices with audio samples.
    """
    if logger:
        logger.info(f"Designing voice: {design_prompt[:80]}")

    body: Dict[str, Any] = {
        "langCode": lang_code,
        "designPrompt": design_prompt,
        "previewText": preview_text,
        "voiceDesignConfig": {
            "numberOfSamples": number_of_samples,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{VOICES_BASE_URL}/voices:design",
            headers={
                "Authorization": get_auth_header(),
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

    result = response.json()
    previews = result.get("previewVoices", [])

    if logger:
        logger.info(f"Generated {len(previews)} voice previews")

    return result


async def publish_voice(
    voice_id: str,
    display_name: str,
    description: Optional[str] = None,
    tags: Optional[list] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Publish a designed/cloned voice to make it permanently available.

    Returns the published voice resource.
    """
    if logger:
        logger.info(f"Publishing voice: {voice_id} as {display_name}")

    body: Dict[str, Any] = {"displayName": display_name}
    if description:
        body["description"] = description
    if tags:
        body["tags"] = tags

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{VOICES_BASE_URL}/voices/{voice_id}:publish",
            headers={
                "Authorization": get_auth_header(),
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

    result = response.json()

    if logger:
        logger.info(f"Voice published: {result.get('voiceId')}")

    return result


async def clone_voice(
    display_name: str,
    audio_data: bytes,
    lang_code: str = "EN_US",
    description: Optional[str] = None,
    remove_background_noise: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Clone a voice from an audio sample using Inworld Voice Cloning API.

    Returns the voice dict with voiceId and metadata.
    """
    if logger:
        logger.info(f"Cloning voice: {display_name}, lang: {lang_code}")

    audio_b64 = base64.b64encode(audio_data).decode()

    body: Dict[str, Any] = {
        "displayName": display_name,
        "langCode": lang_code,
        "voiceSamples": [{"audioData": audio_b64}],
        "audioProcessingConfig": {
            "removeBackgroundNoise": remove_background_noise,
        },
    }
    if description:
        body["description"] = description

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{VOICES_BASE_URL}/voices:clone",
            headers={
                "Authorization": get_auth_header(),
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

    result = response.json()
    voice = result.get("voice", {})

    if logger:
        logger.info(f"Voice cloned: {voice.get('voiceId')} ({voice.get('displayName')})")

    return result


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
