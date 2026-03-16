"""
Shared helper module for ElevenLabs API operations.
Uses the official ElevenLabs SDK for reliable API access.

Symlink this file into your app folder for deployment.

Note: speech_to_text() and create_dubbing() accept URLs directly via
cloud_storage_url/source_url params, but only for publicly accessible URLs.
"""

import os
import logging
import tempfile
import asyncio
import subprocess
import json
from typing import Optional, Dict, Any, List
from io import BytesIO

from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings


# ElevenLabs premade voices - name to ID mapping
VOICE_IDS = {
    "adam": "pNInz6obpgDQGcFmaJgB",        # American male, dominant/firm
    "alice": "Xb7hH8MSUJpSbSDYk0k2",       # British female, clear/engaging
    "aria": "9BWtsMINqrJLrRacOk9x",        # American female, expressive
    "bella": "hpp4J3VqNfWAUOO0d1Us",       # American female, professional/warm
    "bill": "pqHfZKP75CvOlQylNhV4",        # American male, wise/mature
    "brian": "nPczCjzI2devNBz1zQrb",       # American male, deep/comforting
    "callum": "N2lVS1w4EtoT3dr4eOWO",      # American male, husky
    "charlie": "IKne3meq5aSn9XLyUdCD",     # Australian male, deep/energetic
    "chris": "iP95p4xoKVk53GoZ742B",       # American male, charming
    "daniel": "onwK4e9ZLuTAKqWW03F9",      # British male, broadcaster
    "eric": "cjVigY5qzO86Huf0OWal",        # American male, smooth/trustworthy
    "george": "JBFqnCBsd6RMkjVDRZzb",      # British male, warm storyteller
    "harry": "SOYHLrjzK2X1ezoPC6cr",       # American male, fierce/rough
    "jessica": "cgSgspJ2msm6clMCkdW9",     # American female, playful/bright
    "laura": "FGY2WhTYpPnrIDTdsKH5",       # American female, quirky/sassy
    "liam": "TX3LPaxmHKxFdv7VOQHJ",        # American male, energetic
    "lily": "pFZP5JQG7iQjIQuC4Bku",        # British female, velvety
    "matilda": "XrExE9yKIg1WjnnlVkGX",     # American female, professional
    "river": "SAz9YHcvj6GT2YYXdXww",       # American neutral, calm/informative
    "roger": "CwhRBWXzGAHq8TQ4Fs17",       # American male, laid-back
    "sarah": "EXAVITQu4vr4xnSDxMaL",       # American female, confident
    "will": "bIHbv24MWmeRgasZH58o",        # American male, relaxed
}


def get_client() -> ElevenLabs:
    """Get ElevenLabs client with API key from environment."""
    key = os.environ.get("ELEVENLABS_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_KEY environment variable is required")
    return ElevenLabs(api_key=key)


def get_api_key() -> str:
    """Validate API key exists."""
    key = os.environ.get("ELEVENLABS_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_KEY environment variable is required")
    return key


def get_voice_id(voice: str) -> str:
    """Get voice ID from voice name or return as-is if already an ID."""
    return VOICE_IDS.get(voice.lower(), voice)


def save_audio_bytes(audio_bytes: bytes, suffix: str = ".mp3") -> str:
    """Save audio bytes to a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def save_audio_generator(audio_gen, suffix: str = ".mp3") -> str:
    """Save audio generator to a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in audio_gen:
            tmp.write(chunk)
        return tmp.name


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


def text_to_speech(
    text: str,
    voice_id: str,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    voice_settings: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Convert text to speech using ElevenLabs TTS API."""
    if logger:
        logger.info(f"Generating speech with voice: {voice_id}, model: {model_id}")

    client = get_client()
    voice_id = get_voice_id(voice_id)

    # Convert dict to VoiceSettings if provided
    settings = None
    if voice_settings:
        settings = VoiceSettings(**voice_settings)

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format=output_format,
        voice_settings=settings,
    )

    ext = ".mp3" if "mp3" in output_format else ".pcm"
    file_path = save_audio_generator(audio, suffix=ext)

    if logger:
        logger.info(f"Audio saved to: {file_path}")
    return file_path


def speech_to_text(
    audio: str,
    model_id: str = "scribe_v2",
    language_code: Optional[str] = None,
    diarize: bool = False,
    tag_audio_events: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Transcribe audio using ElevenLabs Speech-to-Text API.

    Args:
        audio: File path or URL. If URL, uses cloud_storage_url to avoid upload.
    """
    client = get_client()
    is_url = audio.startswith("http://") or audio.startswith("https://")

    if logger:
        if is_url:
            logger.info(f"Transcribing audio from URL: {audio[:60]}...")
        else:
            logger.info(f"Transcribing audio: {audio}")

    if is_url:
        result = client.speech_to_text.convert(
            cloud_storage_url=audio,
            model_id=model_id,
            language_code=language_code,
            diarize=diarize,
            tag_audio_events=tag_audio_events,
        )
    else:
        with open(audio, "rb") as f:
            result = client.speech_to_text.convert(
                file=f,
                model_id=model_id,
                language_code=language_code,
                diarize=diarize,
                tag_audio_events=tag_audio_events,
            )

    if logger:
        logger.info("Transcription completed")
    return result.model_dump()


def speech_to_speech(
    audio_path: str,
    voice_id: str,
    model_id: str = "eleven_multilingual_sts_v2",
    output_format: str = "mp3_44100_128",
    voice_settings: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Transform voice in audio using ElevenLabs Voice Changer API."""
    if logger:
        logger.info(f"Transforming voice to: {voice_id}")

    client = get_client()
    voice_id = get_voice_id(voice_id)

    # Read file content into BytesIO - SDK needs seekable file-like object
    with open(audio_path, "rb") as f:
        audio_file = BytesIO(f.read())

    audio = client.speech_to_speech.convert(
        voice_id=voice_id,
        audio=audio_file,
        model_id=model_id,
        output_format=output_format,
    )

    file_path = save_audio_generator(audio, suffix=".mp3")

    if logger:
        logger.info(f"Transformed audio saved to: {file_path}")
    return file_path


def isolate_voice(
    audio_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Remove background noise from audio using Voice Isolator API."""
    if logger:
        logger.info(f"Isolating voice from: {audio_path}")

    client = get_client()

    # Read file content into BytesIO
    with open(audio_path, "rb") as f:
        audio_file = BytesIO(f.read())

    audio = client.audio_isolation.convert(audio=audio_file)

    file_path = save_audio_generator(audio, suffix=".mp3")

    if logger:
        logger.info(f"Isolated audio saved to: {file_path}")
    return file_path


def generate_sound_effect(
    text: str,
    duration_seconds: Optional[float] = None,
    prompt_influence: float = 0.3,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Generate sound effects from text description."""
    if logger:
        logger.info(f"Generating sound effect: {text[:50]}...")

    client = get_client()

    audio = client.text_to_sound_effects.convert(
        text=text,
        duration_seconds=duration_seconds,
        prompt_influence=prompt_influence,
    )

    file_path = save_audio_generator(audio, suffix=".mp3")

    if logger:
        logger.info(f"Sound effect saved to: {file_path}")
    return file_path


def compose_music(
    prompt: str,
    duration_seconds: int = 30,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Generate music from a text prompt."""
    if logger:
        logger.info(f"Composing music: {prompt[:50]}...")

    client = get_client()

    # SDK uses duration in milliseconds
    audio = client.music.compose(
        prompt=prompt,
        music_length_ms=duration_seconds * 1000,
    )

    file_path = save_audio_generator(audio, suffix=".mp3")

    if logger:
        logger.info(f"Music saved to: {file_path}")
    return file_path


async def create_dubbing(
    audio: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Dub audio to another language.

    Args:
        audio: File path or URL. If URL, uses source_url to avoid upload.
    """
    if logger:
        logger.info(f"Creating dubbing to {target_lang}")

    client = get_client()
    is_url = audio.startswith("http://") or audio.startswith("https://")

    if is_url:
        if logger:
            logger.info(f"Using source URL: {audio[:60]}...")
        result = client.dubbing.create(
            source_url=audio,
            target_lang=target_lang,
            source_lang=source_lang,
        )
    else:
        # Dubbing requires file with proper name/type
        filename = os.path.basename(audio)
        with open(audio, "rb") as f:
            audio_content = f.read()

        # Create a named BytesIO-like object
        audio_file = BytesIO(audio_content)
        audio_file.name = filename  # Add filename attribute

        result = client.dubbing.create(
            file=audio_file,
            target_lang=target_lang,
            source_lang=source_lang,
        )

    dubbing_id = result.dubbing_id

    if logger:
        logger.info(f"Dubbing job started: {dubbing_id}")

    # Poll for completion
    dubbed_audio = await poll_dubbing_status(dubbing_id, target_lang, logger=logger)
    return dubbed_audio


async def poll_dubbing_status(
    dubbing_id: str,
    target_lang: str,
    poll_interval: float = 5.0,
    max_attempts: int = 120,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Poll dubbing status until completion."""
    client = get_client()

    for attempt in range(max_attempts):
        result = client.dubbing.get(dubbing_id=dubbing_id)
        status = result.status

        if status == "dubbed":
            if logger:
                logger.info("Dubbing completed, downloading...")
            return download_dubbed_audio(dubbing_id, target_lang, logger=logger)
        elif status == "failed":
            error = getattr(result, "error", "Unknown error")
            raise RuntimeError(f"Dubbing failed: {error}")
        else:
            if logger and attempt % 6 == 0:
                logger.info(f"Dubbing status: {status}, waiting...")
            await asyncio.sleep(poll_interval)

    raise RuntimeError(f"Dubbing timed out after {max_attempts * poll_interval} seconds")


def download_dubbed_audio(
    dubbing_id: str,
    language_code: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Download dubbed audio file."""
    client = get_client()

    audio = client.dubbing.audio.get(
        dubbing_id=dubbing_id,
        language_code=language_code,
    )

    file_path = save_audio_generator(audio, suffix=".mp3")

    if logger:
        logger.info(f"Dubbed audio saved to: {file_path}")
    return file_path


def text_to_dialogue(
    inputs: List[Dict[str, str]],
    logger: Optional[logging.Logger] = None,
) -> str:
    """Generate dialogue audio from multiple text/voice pairs."""
    if logger:
        logger.info(f"Generating dialogue with {len(inputs)} segments")

    client = get_client()

    # Resolve voice names to IDs
    resolved_inputs = [
        {"text": inp["text"], "voice_id": get_voice_id(inp["voice_id"])}
        for inp in inputs
    ]

    audio = client.text_to_dialogue.convert(inputs=resolved_inputs)

    file_path = save_audio_generator(audio, suffix=".mp3")

    if logger:
        logger.info(f"Dialogue audio saved to: {file_path}")
    return file_path


def forced_alignment(
    audio_path: str,
    text: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Align text to audio, returning word-level timestamps."""
    if logger:
        logger.info(f"Performing forced alignment on: {audio_path}")

    client = get_client()

    # Read file into BytesIO
    with open(audio_path, "rb") as f:
        audio_file = BytesIO(f.read())

    result = client.forced_alignment.create(
        file=audio_file,
        text=text,
    )

    if logger:
        logger.info("Forced alignment completed")
    return result.model_dump()
