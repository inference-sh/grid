"""
Common utilities for Google Gemini API models.

This module provides shared functionality for Gemini API integrations:
- Enums for output formats, aspect ratios, resolutions, safety settings
- Client initialization (API key auth)
- Image/video file handling utilities
- Dimension calculations
- Retry with exponential backoff
- Video generation via google-genai SDK
"""

import os
import math
import logging
import tempfile
import asyncio
import base64
import io
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Callable, TypeVar, Awaitable

from PIL import Image

from google import genai
from google.genai import types

T = TypeVar('T')


# =============================================================================
# ENUMS
# =============================================================================

class OutputFormatEnum(str, Enum):
    """Output format options for images."""
    png = "png"
    jpeg = "jpeg"
    webp = "webp"


class OutputFormatExtendedEnum(str, Enum):
    """Extended output format options (includes HEIC/HEIF)."""
    png = "png"
    jpeg = "jpeg"
    webp = "webp"
    heic = "heic"
    heif = "heif"


class AspectRatioEnum(str, Enum):
    """Aspect ratio options."""
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_3_2 = "3:2"
    ratio_4_3 = "4:3"
    ratio_5_4 = "5:4"
    ratio_1_1 = "1:1"
    ratio_4_5 = "4:5"
    ratio_3_4 = "3:4"
    ratio_2_3 = "2:3"
    ratio_9_16 = "9:16"


class AspectRatioAutoEnum(str, Enum):
    """Aspect ratio options with auto detection."""
    auto = "auto"
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_3_2 = "3:2"
    ratio_4_3 = "4:3"
    ratio_5_4 = "5:4"
    ratio_1_1 = "1:1"
    ratio_4_5 = "4:5"
    ratio_3_4 = "3:4"
    ratio_2_3 = "2:3"
    ratio_9_16 = "9:16"


class SafetyToleranceEnum(str, Enum):
    """Safety filter thresholds."""
    block_none = "BLOCK_NONE"
    block_low_and_above = "BLOCK_LOW_AND_ABOVE"
    block_medium_and_above = "BLOCK_MEDIUM_AND_ABOVE"
    block_only_high = "BLOCK_ONLY_HIGH"


class ResolutionEnum(str, Enum):
    """Resolution options for images."""
    res_512 = "512"
    res_1k = "1K"
    res_2k = "2K"
    res_4k = "4K"


class VideoAspectRatioEnum(str, Enum):
    """Aspect ratio options for video (16:9 and 9:16)."""
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"


class VideoResolutionEnum(str, Enum):
    """Resolution options for video."""
    res_720p = "720p"
    res_1080p = "1080p"
    res_4k = "4k"


class PersonGenerationEnum(str, Enum):
    """Person generation settings for video."""
    allow_adult = "allow_adult"
    disallow = "disallow"


# =============================================================================
# CONSTANTS
# =============================================================================

ASPECT_RATIO_VALUES = {
    "8:1": 8/1,
    "21:9": 21/9,
    "16:9": 16/9,
    "4:1": 4/1,
    "3:2": 3/2,
    "4:3": 4/3,
    "5:4": 5/4,
    "1:1": 1.0,
    "4:5": 4/5,
    "3:4": 3/4,
    "2:3": 2/3,
    "1:4": 1/4,
    "9:16": 9/16,
    "1:8": 1/8,
}

IMAGE_MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
    '.gif': 'image/gif',
}

VIDEO_MIME_TYPES = {
    '.mp4': 'video/mp4',
    '.mpeg': 'video/mpeg',
    '.mov': 'video/quicktime',
    '.avi': 'video/x-msvideo',
    '.webm': 'video/webm',
    '.mkv': 'video/x-matroska',
    '.flv': 'video/x-flv',
    '.wmv': 'video/x-ms-wmv',
    '.3gp': 'video/3gpp',
}

SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT"
]


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """
    Configuration for exponential backoff retry on 429 RESOURCE_EXHAUSTED errors.

    Delay schedule with defaults (base_delay_ms=300, multiplier=2, max_attempts=6):
        Attempt 1: 0-300ms
        Attempt 2: 0-600ms
        Attempt 3: 0-1.2s
        Attempt 4: 0-2.4s
        Attempt 5: 0-4.8s
        Attempt 6: 0-9.6s
    """
    max_attempts: int = 6
    base_delay_ms: float = 300.0
    multiplier: float = 2.0

    def get_max_delay_ms(self, attempt: int) -> float:
        return self.base_delay_ms * (self.multiplier ** (attempt - 1))

    def get_jittered_delay_s(self, attempt: int) -> float:
        max_delay_ms = self.get_max_delay_ms(attempt)
        delay_ms = random.uniform(0, max_delay_ms)
        return delay_ms / 1000.0


DEFAULT_RETRY_CONFIG = RetryConfig()
_retry_logger = logging.getLogger(__name__ + ".retry")


def is_resource_exhausted_error(error: Exception) -> bool:
    error_str = str(error)
    return "429" in error_str and "RESOURCE_EXHAUSTED" in error_str


async def retry_on_resource_exhausted(
    func: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None,
    logger: Optional[logging.Logger] = None
) -> T:
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    log = logger or _retry_logger
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func()
        except Exception as e:
            if is_resource_exhausted_error(e):
                last_exception = e
                if attempt < config.max_attempts:
                    delay_s = config.get_jittered_delay_s(attempt)
                    max_delay_ms = config.get_max_delay_ms(attempt)
                    log.warning(
                        f"429 RESOURCE_EXHAUSTED on attempt {attempt}/{config.max_attempts}. "
                        f"Retrying in {delay_s*1000:.0f}ms (max: {max_delay_ms:.0f}ms)"
                    )
                    await asyncio.sleep(delay_s)
                else:
                    log.error(f"429 RESOURCE_EXHAUSTED: Max retries ({config.max_attempts}) exceeded")
                    raise
            else:
                raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_on_resource_exhausted")


# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

def create_gemini_client() -> genai.Client:
    """
    Create a Gemini API client using API key from environment.

    Requires: GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for model access.")
    return genai.Client(api_key=api_key)


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def get_mime_type(file_path: str, default: str = 'application/octet-stream') -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_MIME_TYPES:
        return IMAGE_MIME_TYPES[ext]
    if ext in VIDEO_MIME_TYPES:
        return VIDEO_MIME_TYPES[ext]
    return default


def get_image_dimensions(file_path: str) -> tuple[int, int]:
    with Image.open(file_path) as img:
        return img.size


def find_closest_aspect_ratio(width: int, height: int) -> str:
    if height == 0:
        return "1:1"
    actual_ratio = width / height
    closest_ratio = "1:1"
    min_diff = float('inf')
    for ratio_str, ratio_val in ASPECT_RATIO_VALUES.items():
        diff = abs(actual_ratio - ratio_val)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = ratio_str
    return closest_ratio


def calculate_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    base = 1024
    if resolution == "512":
        base = 512
    elif resolution == "2K":
        base = 2048
    elif resolution == "4K":
        base = 4096

    ratio_val = ASPECT_RATIO_VALUES.get(aspect_ratio, 1.0)
    height = int(base / math.sqrt(ratio_val))
    width = int(height * ratio_val)
    return width, height


def resize_image_to_max_pixels(
    file_path: str,
    max_pixels: int = 1_000_000,
    logger: Optional[logging.Logger] = None
) -> bytes:
    with Image.open(file_path) as img:
        original_width, original_height = img.size
        current_pixels = original_width * original_height

        if current_pixels <= max_pixels:
            with open(file_path, 'rb') as f:
                return f.read()

        scale = math.sqrt(max_pixels / current_pixels)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        if logger:
            logger.info(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")

        img = img.convert("RGB") if img.mode != "RGB" else img
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            resized.save(output, format="JPEG", quality=95)
        elif ext == '.webp':
            resized.save(output, format="WEBP", quality=95)
        else:
            resized.save(output, format="PNG")

        return output.getvalue()


def resolve_aspect_ratio(
    aspect_ratio_value: str,
    images: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    if aspect_ratio_value != "auto":
        return aspect_ratio_value

    if images and len(images) > 0:
        first_image_path = images[0].path
        img_width, img_height = get_image_dimensions(first_image_path)
        resolved = find_closest_aspect_ratio(img_width, img_height)
        if logger:
            logger.info(f"Auto-detected aspect ratio: {resolved} (from {img_width}x{img_height})")
        return resolved
    else:
        if logger:
            logger.info("No input images for auto aspect ratio detection, using 1:1")
        return "1:1"


def load_image_as_part(
    file_path: str,
    max_pixels: int = 1_000_000,
    logger: Optional[logging.Logger] = None
) -> types.Part:
    image_data = resize_image_to_max_pixels(file_path, max_pixels, logger)
    mime_type = get_mime_type(file_path, default='image/png')
    return types.Part.from_bytes(data=image_data, mime_type=mime_type)


def save_image_to_temp(image_bytes: bytes, output_format: str = "png") -> str:
    file_extension = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
        tmp_file.write(image_bytes)
        return tmp_file.name


# =============================================================================
# IMAGE RESPONSE PROCESSING
# =============================================================================

@dataclass
class ImageResponseResult:
    """Result from processing a Gemini image generation response."""
    image_paths: list = field(default_factory=list)
    descriptions: list = field(default_factory=list)
    finish_reason: Optional[str] = None
    finish_message: Optional[str] = None
    usage_metadata: Optional[Any] = None


def process_image_response(
    response,
    output_format: str = "png",
    logger: Optional[logging.Logger] = None,
) -> ImageResponseResult:
    log = logger or logging.getLogger(__name__)

    if not response.candidates or len(response.candidates) == 0:
        log.error("No candidates in response")
        block_reason = getattr(response, 'prompt_feedback', None)
        if block_reason:
            raise RuntimeError(f"Image generation blocked: {block_reason}")
        raise RuntimeError("No candidates returned from model")

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, 'finish_reason', None)
    finish_message = getattr(candidate, 'finish_message', None)
    usage_metadata = getattr(response, 'usage_metadata', None)

    log.info(f"Finish reason: {finish_reason}")
    if finish_message:
        log.info(f"Finish message: {finish_message}")
    if usage_metadata:
        log.info(f"Usage metadata: {usage_metadata}")

    result = ImageResponseResult(
        finish_reason=str(finish_reason) if finish_reason else None,
        finish_message=finish_message,
        usage_metadata=usage_metadata,
    )

    parts = getattr(candidate.content, 'parts', None) if candidate.content else None
    for part in (parts or []):
        if hasattr(part, 'thought') and part.thought:
            continue

        if part.inline_data is not None:
            image_path = save_image_to_temp(part.inline_data.data, output_format)
            result.image_paths.append(image_path)
            log.info(f"Saved image to {image_path}")
        elif part.text is not None and part.text.strip():
            result.descriptions.append(part.text)
            log.info(f"Model response: {part.text[:200]}...")

    return result


def raise_no_images_error(results: list) -> None:
    error_parts = ["No images were generated"]
    if results:
        last = results[-1]
        if last.finish_reason:
            error_parts.append(f"finish_reason={last.finish_reason}")
        if last.finish_message:
            error_parts.append(last.finish_message)
        all_descs = []
        for r in results:
            all_descs.extend(r.descriptions)
        desc = "\n".join(all_descs).strip()
        if desc:
            error_parts.append(f"Model response: {desc[:500]}")
    raise RuntimeError(". ".join(error_parts))


def build_image_output_meta(results: list, width: int = 0, height: int = 0):
    inputs = []
    outputs = []
    total_images = 0

    for r in results:
        total_images += len(r.image_paths)
        usage_metadata = getattr(r, 'usage_metadata', None)
        if usage_metadata:
            um = usage_metadata
            print(um)
            prompt_tokens = getattr(um, 'prompt_token_count', None)
            if prompt_tokens:
                inputs.append({"type": "text", "tokens": prompt_tokens})
            thoughts_tokens = getattr(um, 'thoughts_token_count', None)
            if thoughts_tokens:
                outputs.append({"type": "text", "tokens": thoughts_tokens})
            candidates_tokens = getattr(um, 'candidates_token_count', None)
            if candidates_tokens:
                outputs.append({"type": "text", "tokens": candidates_tokens})

    for _ in range(total_images):
        outputs.append({"type": "image", "width": width, "height": height})

    return {"inputs": inputs, "outputs": outputs}


# =============================================================================
# GENERATION CONFIG BUILDERS
# =============================================================================

def build_safety_settings(tolerance: str) -> list:
    return [
        types.SafetySetting(category=category, threshold=tolerance)
        for category in SAFETY_CATEGORIES
    ]


def build_image_generation_config(
    aspect_ratio: str,
    resolution: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    max_output_tokens: int = 32768,
    safety_tolerance: str = "BLOCK_MEDIUM_AND_ABOVE",
    enable_google_search: bool = False,
    response_modalities: list = None,
) -> types.GenerateContentConfig:
    if response_modalities is None:
        response_modalities = ['TEXT', 'IMAGE']

    image_config = types.ImageConfig(
        aspect_ratio=aspect_ratio,
        image_size=resolution,
    )

    config_kwargs = {
        'response_modalities': response_modalities,
        'image_config': image_config,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'max_output_tokens': max_output_tokens,
        'safety_settings': build_safety_settings(safety_tolerance),
    }

    if enable_google_search:
        config_kwargs['tools'] = [{"google_search": {}}]

    return types.GenerateContentConfig(**config_kwargs)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level)
    return logging.getLogger(name)


# =============================================================================
# VIDEO UTILITIES
# =============================================================================

def detect_video_aspect_ratio(width: int, height: int) -> str:
    return "9:16" if height > width else "16:9"


def resize_image_for_video(
    image_bytes: bytes,
    aspect_ratio: str,
    max_dimension: int = 1280
) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))

    if aspect_ratio == "16:9":
        target_width = max_dimension
        target_height = int(max_dimension * 9 / 16)
    else:
        target_height = max_dimension
        target_width = int(max_dimension * 9 / 16)

    img = img.convert("RGB")
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        new_height = target_height
        new_width = int(img_ratio * new_height)
    else:
        new_width = target_width
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    return output.getvalue()


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_base64_to_bytes(base64_string: str) -> bytes:
    return base64.b64decode(base64_string)


def save_video_to_temp(video_bytes: bytes, output_format: str = "mp4") -> str:
    file_extension = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        return tmp_file.name


async def generate_video_with_polling(
    client: genai.Client,
    model_id: str,
    prompt: str,
    image_path: Optional[str] = None,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 8,
    generate_audio: bool = False,
    person_generation: str = "allow_adult",
    poll_interval: float = 5.0,
    max_wait_time: float = 600.0,
    logger: Optional[logging.Logger] = None,
) -> list:
    """
    Generate video using google-genai SDK with polling.

    Uses client.models.generate_videos() and polls the operation until complete.

    Args:
        client: Gemini API client
        model_id: Model ID (e.g., "veo-3.1-lite-generate-preview")
        prompt: Text prompt
        image_path: Optional first frame image path
        aspect_ratio: "16:9" or "9:16"
        duration_seconds: Video duration
        generate_audio: Whether to generate audio
        person_generation: Person generation setting
        poll_interval: Seconds between polls
        max_wait_time: Max wait time in seconds
        logger: Optional logger

    Returns:
        List of generated video objects from the API response
    """
    log = logger or logging.getLogger(__name__)

    # Build the image if provided
    image = None
    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        resized = resize_image_for_video(image_bytes, aspect_ratio)
        image = types.Image(image_bytes=resized, mime_type="image/jpeg")

    config = types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        generate_audio=generate_audio,
        person_generation=person_generation,
    )

    log.info(f"Starting video generation: model={model_id}, aspect_ratio={aspect_ratio}, duration={duration_seconds}s, audio={generate_audio}")

    # Start the operation
    operation = client.models.generate_videos(
        model=model_id,
        prompt=prompt,
        image=image,
        config=config,
    )

    log.info(f"Operation started: {operation.name}")

    # Poll until done
    elapsed = 0.0
    while not operation.done and elapsed < max_wait_time:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        operation = client.operations.get(operation)
        log.info(f"Operation in progress... (elapsed: {elapsed:.0f}s)")

    if not operation.done:
        raise RuntimeError(f"Video generation timed out after {max_wait_time}s")

    if operation.error:
        raise RuntimeError(f"Video generation failed: {operation.error}")

    result = operation.result
    if not result or not result.generated_videos:
        raise RuntimeError("No videos in response")

    log.info(f"Video generation complete: {len(result.generated_videos)} video(s)")
    return result.generated_videos
