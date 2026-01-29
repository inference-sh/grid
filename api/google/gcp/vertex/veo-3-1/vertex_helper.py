"""
Common utilities for Google Vertex AI image/video models.

This module provides shared functionality for Vertex AI model integrations:
- Enums for output formats, aspect ratios, resolutions, safety settings
- Client initialization
- Image/video file handling utilities
- Dimension calculations
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
import aiohttp

T = TypeVar('T')

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.oauth2.credentials import Credentials


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
    off = "OFF"


class ResolutionEnum(str, Enum):
    """Resolution options."""
    res_1k = "1K"
    res_2k = "2K"
    res_4k = "4K"


class VideoAspectRatioEnum(str, Enum):
    """Aspect ratio options for video (Veo supports 16:9 and 9:16)."""
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"


class VideoResolutionEnum(str, Enum):
    """Resolution options for video."""
    res_720p = "720p"
    res_1080p = "1080p"


# =============================================================================
# CONSTANTS
# =============================================================================

# Aspect ratio numeric values for matching
ASPECT_RATIO_VALUES = {
    "21:9": 21/9,
    "16:9": 16/9,
    "3:2": 3/2,
    "4:3": 4/3,
    "5:4": 5/4,
    "1:1": 1.0,
    "4:5": 4/5,
    "3:4": 3/4,
    "2:3": 2/3,
    "9:16": 9/16,
}

# MIME types for image formats
IMAGE_MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
    '.gif': 'image/gif',
}

# MIME types for video formats
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

# Safety categories for Gemini models
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

    The delay uses jitter: actual delay is random between 0 and max_delay for each attempt.
    max_delay = base_delay_ms * (multiplier ** (attempt - 1))
    """
    max_attempts: int = 6
    base_delay_ms: float = 300.0  # Base delay in milliseconds
    multiplier: float = 2.0  # Delay multiplier per attempt

    def get_max_delay_ms(self, attempt: int) -> float:
        """Calculate max delay for a given attempt number (1-indexed)."""
        return self.base_delay_ms * (self.multiplier ** (attempt - 1))

    def get_jittered_delay_s(self, attempt: int) -> float:
        """Get a random delay between 0 and max_delay for the given attempt."""
        max_delay_ms = self.get_max_delay_ms(attempt)
        delay_ms = random.uniform(0, max_delay_ms)
        return delay_ms / 1000.0


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


def is_resource_exhausted_error(error: Exception) -> bool:
    """
    Check if an exception is a 429 RESOURCE_EXHAUSTED error.

    Args:
        error: The exception to check

    Returns:
        True if it's a 429 RESOURCE_EXHAUSTED error
    """
    error_str = str(error)
    return "429" in error_str and "RESOURCE_EXHAUSTED" in error_str


async def retry_on_resource_exhausted(
    func: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None,
    logger: Optional[logging.Logger] = None
) -> T:
    """
    Execute an async function with exponential backoff retry on 429 errors.

    Only retries on 429 RESOURCE_EXHAUSTED errors. All other errors are raised immediately.
    Uses jitter: delay is random between 0 and max_delay for each attempt.

    Args:
        func: Async callable to execute (should be a zero-argument lambda or partial)
        config: Optional RetryConfig, uses DEFAULT_RETRY_CONFIG if not provided
        logger: Optional logger for retry messages

    Returns:
        The result of func() on success

    Raises:
        The original exception if max retries exceeded or non-retryable error

    Example:
        result = await retry_on_resource_exhausted(
            lambda: client.models.generate_content(model=model_id, contents=contents, config=config),
            config=RetryConfig(max_attempts=6, base_delay_ms=300),
            logger=self.logger
        )
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

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

                    if logger:
                        logger.warning(
                            f"429 RESOURCE_EXHAUSTED on attempt {attempt}/{config.max_attempts}. "
                            f"Retrying in {delay_s*1000:.0f}ms (max: {max_delay_ms:.0f}ms)"
                        )

                    await asyncio.sleep(delay_s)
                else:
                    if logger:
                        logger.error(
                            f"429 RESOURCE_EXHAUSTED: Max retries ({config.max_attempts}) exceeded"
                        )
                    raise
            else:
                # Not a 429 error, don't retry
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception

    raise RuntimeError("Unexpected state in retry_on_resource_exhausted")


# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

def create_vertex_client(
    location: Optional[str] = None,
    api_version: str = "v1"
) -> genai.Client:
    """
    Create a Vertex AI client using environment credentials.

    Requires environment variables:
    - GCP_ACCESS_TOKEN: OAuth access token for authentication
    - GCP_PROJECT_NUMBER: GCP project number/ID

    Args:
        location: Optional GCP region (e.g., "us-central1"). None for global.
        api_version: API version to use (default: "v1")

    Returns:
        Configured genai.Client instance

    Raises:
        RuntimeError: If required environment variables are missing
    """
    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    project = os.environ.get("GCP_PROJECT_NUMBER")

    if not access_token:
        raise RuntimeError(
            "GCP_ACCESS_TOKEN environment variable is required for Vertex AI access."
        )
    if not project:
        raise RuntimeError(
            "GCP_PROJECT_NUMBER environment variable is required for Vertex AI access."
        )

    credentials = Credentials(token=access_token)

    client_kwargs = {
        'vertexai': True,
        'project': project,
        'credentials': credentials,
        'http_options': HttpOptions(api_version=api_version)
    }

    if location:
        client_kwargs['location'] = location

    return genai.Client(**client_kwargs)


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def get_mime_type(file_path: str, default: str = 'application/octet-stream') -> str:
    """
    Get MIME type based on file extension.

    Args:
        file_path: Path to the file
        default: Default MIME type if extension not recognized

    Returns:
        MIME type string
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_MIME_TYPES:
        return IMAGE_MIME_TYPES[ext]
    if ext in VIDEO_MIME_TYPES:
        return VIDEO_MIME_TYPES[ext]

    return default


def get_image_dimensions(file_path: str) -> tuple[int, int]:
    """
    Get width and height of an image file using PIL.

    Args:
        file_path: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    with Image.open(file_path) as img:
        return img.size


def find_closest_aspect_ratio(width: int, height: int) -> str:
    """
    Find the closest supported aspect ratio for given dimensions.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Closest aspect ratio string (e.g., "16:9")
    """
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
    """
    Calculate pixel dimensions from aspect ratio and resolution.

    Uses equal-area approximation where w * h = base^2.

    Args:
        aspect_ratio: Aspect ratio string (e.g., "16:9")
        resolution: Resolution string ("1K", "2K", or "4K")

    Returns:
        Tuple of (width, height)
    """
    base = 1024
    if resolution == "2K":
        base = 2048
    elif resolution == "4K":
        base = 4096

    ratio_val = ASPECT_RATIO_VALUES.get(aspect_ratio, 1.0)

    # Equal area: w * h = base^2, w/h = ratio
    # h = base / sqrt(ratio)
    height = int(base / math.sqrt(ratio_val))
    width = int(height * ratio_val)

    return width, height


def resize_image_to_max_pixels(
    file_path: str,
    max_pixels: int = 1_000_000,
    logger: Optional[logging.Logger] = None
) -> bytes:
    """
    Resize an image to fit within max_pixels while preserving aspect ratio.

    Args:
        file_path: Path to the image file
        max_pixels: Maximum total pixels (default: 1MP = 1,000,000)
        logger: Optional logger

    Returns:
        Resized image as bytes in original format
    """
    with Image.open(file_path) as img:
        original_width, original_height = img.size
        current_pixels = original_width * original_height

        if current_pixels <= max_pixels:
            # No resize needed, return original bytes
            with open(file_path, 'rb') as f:
                return f.read()

        # Calculate scale factor to fit within max_pixels
        scale = math.sqrt(max_pixels / current_pixels)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        if logger:
            logger.info(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")

        # Resize the image
        img = img.convert("RGB") if img.mode != "RGB" else img
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save to bytes
        output = io.BytesIO()
        # Determine format from file extension
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
    """
    Resolve aspect ratio, handling 'auto' by detecting from first image.

    Args:
        aspect_ratio_value: Aspect ratio string (may be "auto")
        images: Optional list of image File objects
        logger: Optional logger

    Returns:
        Resolved aspect ratio string (e.g., "16:9")
    """
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
    """
    Load an image file and return it as a Gemini Part.
    Resizes to max_pixels if needed.

    Args:
        file_path: Path to the image file
        max_pixels: Maximum total pixels (default: 1MP)
        logger: Optional logger

    Returns:
        Gemini types.Part with image data
    """
    image_data = resize_image_to_max_pixels(file_path, max_pixels, logger)
    mime_type = get_mime_type(file_path, default='image/png')
    return types.Part.from_bytes(data=image_data, mime_type=mime_type)


def load_video_as_part(file_path: str) -> types.Part:
    """
    Load a video file and return it as a Gemini Part.

    Args:
        file_path: Path to the video file

    Returns:
        Gemini types.Part with video data
    """
    with open(file_path, 'rb') as f:
        video_data = f.read()

    mime_type = get_mime_type(file_path, default='video/mp4')
    return types.Part.from_bytes(data=video_data, mime_type=mime_type)


def save_image_to_temp(
    image_bytes: bytes,
    output_format: str = "png"
) -> str:
    """
    Save image bytes to a temporary file.

    Args:
        image_bytes: Raw image data
        output_format: Output format (png, jpeg, webp, etc.)

    Returns:
        Path to the saved temporary file
    """
    file_extension = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
        tmp_file.write(image_bytes)
        return tmp_file.name


# =============================================================================
# GENERATION CONFIG BUILDERS
# =============================================================================

def build_safety_settings(tolerance: str) -> list:
    """
    Build safety settings for all harm categories.

    Args:
        tolerance: Safety tolerance level (e.g., "BLOCK_MEDIUM_AND_ABOVE")

    Returns:
        List of SafetySetting objects
    """
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
    response_modalities: list = None
) -> types.GenerateContentConfig:
    """
    Build a GenerateContentConfig for image generation.

    Args:
        aspect_ratio: Output aspect ratio
        resolution: Output resolution (1K, 2K, 4K)
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum output tokens
        safety_tolerance: Safety filter threshold
        enable_google_search: Whether to enable Google Search grounding
        response_modalities: List of response modalities (default: ['TEXT', 'IMAGE'])

    Returns:
        Configured GenerateContentConfig
    """
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
    """
    Set up a logger with standard configuration.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=level)
    return logging.getLogger(name)


# =============================================================================
# VERTEX AI REST API (for Veo and other long-running operations)
# =============================================================================

def get_vertex_credentials() -> Tuple[str, str]:
    """
    Get Vertex AI credentials from environment.

    Returns:
        Tuple of (access_token, project_id)

    Raises:
        RuntimeError: If required environment variables are missing
    """
    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    project = os.environ.get("GCP_PROJECT_NUMBER")

    if not access_token:
        raise RuntimeError(
            "GCP_ACCESS_TOKEN environment variable is required for Vertex AI access."
        )
    if not project:
        raise RuntimeError(
            "GCP_PROJECT_NUMBER environment variable is required for Vertex AI access."
        )

    return access_token, project


def get_vertex_api_url(
    project: str,
    location: str,
    model_id: str,
    endpoint: str = "predictLongRunning"
) -> str:
    """
    Build Vertex AI API URL.

    Args:
        project: GCP project ID
        location: GCP region (e.g., "us-central1")
        model_id: Model ID (e.g., "veo-3.1-fast-generate-001")
        endpoint: API endpoint (default: "predictLongRunning")

    Returns:
        Full API URL
    """
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/"
        f"publishers/google/models/{model_id}:{endpoint}"
    )


async def start_long_running_operation(
    access_token: str,
    project: str,
    location: str,
    model_id: str,
    payload: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Start a long-running Vertex AI operation (e.g., video generation).

    Args:
        access_token: GCP access token
        project: GCP project ID
        location: GCP region
        model_id: Model ID
        payload: Request payload
        logger: Optional logger

    Returns:
        Operation response with operation name

    Raises:
        RuntimeError: If the request fails
    """
    url = get_vertex_api_url(project, location, model_id, "predictLongRunning")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if logger:
                    logger.info(f"Started operation: {result.get('name', 'unknown')}")
                return result
            else:
                error_text = await response.text()
                if logger:
                    logger.error(f"Error starting operation: {response.status}")
                    logger.error(f"Response: {error_text}")
                raise RuntimeError(f"Failed to start operation: {response.status} - {error_text}")


async def poll_long_running_operation(
    access_token: str,
    project: str,
    location: str,
    model_id: str,
    operation_name: str,
    poll_interval: float = 5.0,
    max_wait_time: float = 600.0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Poll a long-running operation until completion.

    Args:
        access_token: GCP access token
        project: GCP project ID
        location: GCP region
        model_id: Model ID
        operation_name: Full operation name from start response
        poll_interval: Seconds between polls (default: 5)
        max_wait_time: Maximum wait time in seconds (default: 600)
        logger: Optional logger

    Returns:
        Final operation response with results

    Raises:
        RuntimeError: If operation fails or times out
    """
    url = get_vertex_api_url(project, location, model_id, "fetchPredictOperation")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    payload = {"operationName": operation_name}

    elapsed = 0.0
    async with aiohttp.ClientSession() as session:
        while elapsed < max_wait_time:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Poll failed: {response.status} - {error_text}")

                result = await response.json()

                if result.get("done"):
                    if logger:
                        logger.info("Operation completed successfully")
                    return result

                if logger:
                    logger.info(f"Operation in progress... (elapsed: {elapsed:.0f}s)")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    raise RuntimeError(f"Operation timed out after {max_wait_time}s")


# =============================================================================
# VIDEO UTILITIES (for Veo)
# =============================================================================

def detect_video_aspect_ratio(width: int, height: int) -> str:
    """
    Detect video aspect ratio from dimensions (16:9 or 9:16 only).

    Args:
        width: Image/video width
        height: Image/video height

    Returns:
        "16:9" for landscape, "9:16" for portrait
    """
    return "9:16" if height > width else "16:9"


def resize_image_for_video(
    image_bytes: bytes,
    aspect_ratio: str,
    max_dimension: int = 1280
) -> bytes:
    """
    Resize image to match video aspect ratio requirements.

    Args:
        image_bytes: Raw image data
        aspect_ratio: Target aspect ratio ("16:9" or "9:16")
        max_dimension: Maximum dimension (default: 1280)

    Returns:
        Resized image as JPEG bytes
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Calculate target dimensions
    if aspect_ratio == "16:9":
        target_width = max_dimension
        target_height = int(max_dimension * 9 / 16)
    else:  # 9:16
        target_height = max_dimension
        target_width = int(max_dimension * 9 / 16)

    # Resize maintaining aspect ratio, then crop/pad to exact size
    img = img.convert("RGB")

    # Calculate scaling to cover target
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Image is wider, scale by height
        new_height = target_height
        new_width = int(img_ratio * new_height)
    else:
        # Image is taller, scale by width
        new_width = target_width
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop to target size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))

    # Convert to JPEG bytes
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    return output.getvalue()


def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image data

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_base64_to_bytes(base64_string: str) -> bytes:
    """
    Decode base64 string to bytes.

    Args:
        base64_string: Base64-encoded string

    Returns:
        Decoded bytes
    """
    return base64.b64decode(base64_string)


def prepare_image_for_veo(file_path: str, aspect_ratio: str) -> Dict[str, str]:
    """
    Prepare an image file for Veo API (resize and encode).

    Args:
        file_path: Path to image file
        aspect_ratio: Target aspect ratio ("16:9" or "9:16")

    Returns:
        Dict with bytesBase64Encoded and mimeType
    """
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    resized = resize_image_for_video(image_bytes, aspect_ratio)
    encoded = encode_image_to_base64(resized)

    return {
        "bytesBase64Encoded": encoded,
        "mimeType": "image/jpeg"
    }


def build_veo_payload(
    prompt: str,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 8,
    resolution: str = "720p",
    generate_audio: bool = False,
    sample_count: int = 1,
    first_frame_path: Optional[str] = None,
    last_frame_path: Optional[str] = None,
    storage_uri: Optional[str] = None,
    person_generation: str = "allow_all",
    enable_prompt_rewriting: bool = True,
    add_watermark: bool = False,
) -> Dict[str, Any]:
    """
    Build request payload for Veo video generation.

    Args:
        prompt: Text prompt for video generation
        aspect_ratio: "16:9" or "9:16"
        duration_seconds: Video duration (default: 8)
        resolution: "720p" or "1080p"
        generate_audio: Whether to generate audio
        sample_count: Number of videos to generate (1-2)
        first_frame_path: Optional path to first frame image
        last_frame_path: Optional path to last frame image
        storage_uri: Optional GCS URI for output (e.g., "gs://bucket/path/")
        person_generation: Person generation setting
        enable_prompt_rewriting: Whether to allow prompt rewriting
        add_watermark: Whether to add watermark

    Returns:
        Request payload dict
    """
    instance: Dict[str, Any] = {"prompt": prompt}

    # Add first frame if provided
    if first_frame_path:
        instance["image"] = prepare_image_for_veo(first_frame_path, aspect_ratio)

    # Add last frame if provided
    if last_frame_path:
        instance["lastFrame"] = prepare_image_for_veo(last_frame_path, aspect_ratio)

    parameters: Dict[str, Any] = {
        "aspectRatio": aspect_ratio,
        "sampleCount": sample_count,
        "durationSeconds": str(duration_seconds),
        "resolution": resolution,
        "generateAudio": generate_audio,
        "personGeneration": person_generation,
        "enablePromptRewriting": enable_prompt_rewriting,
        "addWatermark": add_watermark,
        "includeRaiReason": False,
    }

    if storage_uri:
        parameters["storageUri"] = storage_uri

    return {
        "instances": [instance],
        "parameters": parameters
    }


async def download_video_from_gcs(
    gcs_uri: str,
    access_token: str,
    logger: Optional[logging.Logger] = None
) -> bytes:
    """
    Download video from Google Cloud Storage.

    Args:
        gcs_uri: GCS URI (e.g., "gs://bucket/path/video.mp4")
        access_token: GCP access token
        logger: Optional logger

    Returns:
        Video bytes

    Raises:
        RuntimeError: If download fails
    """
    # Parse GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    path = gcs_uri[5:]  # Remove "gs://"
    bucket, *object_parts = path.split("/")
    object_name = "/".join(object_parts)

    # Use GCS JSON API
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{object_name.replace('/', '%2F')}?alt=media"

    headers = {"Authorization": f"Bearer {access_token}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                if logger:
                    logger.info(f"Downloaded video from {gcs_uri}")
                return await response.read()
            else:
                error_text = await response.text()
                raise RuntimeError(f"Failed to download from GCS: {response.status} - {error_text}")


def save_video_to_temp(video_bytes: bytes, output_format: str = "mp4") -> str:
    """
    Save video bytes to a temporary file.

    Args:
        video_bytes: Raw video data
        output_format: Output format (default: "mp4")

    Returns:
        Path to the saved temporary file
    """
    file_extension = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        return tmp_file.name
