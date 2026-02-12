"""
Common utilities for xAI image/video models.

This module provides shared functionality for xAI model integrations:
- Aspect ratio types and dimension maps
- Client initialization
- Image encoding and saving utilities
- Auto aspect ratio detection from input images
"""

import asyncio
import base64
import os
import logging
import random
import tempfile
import math
from dataclasses import dataclass
from typing import Optional, Literal, Callable, TypeVar

from inferencesh import File
from xai_sdk import Client

T = TypeVar('T')


# =============================================================================
# ASPECT RATIO TYPES
# =============================================================================

AspectRatioType = Literal[
    "1:1",
    "16:9", "9:16",
    "4:3", "3:4",
    "3:2", "2:3",
    "2:1", "1:2",
    "19.5:9", "9:19.5",
    "20:9", "9:20",
]

AspectRatioAutoType = Literal[
    "auto",
    "1:1",
    "16:9", "9:16",
    "4:3", "3:4",
    "3:2", "2:3",
    "2:1", "1:2",
    "19.5:9", "9:19.5",
    "20:9", "9:20",
]

VideoAspectRatioType = Literal[
    "16:9", "9:16",
    "4:3", "3:4",
    "3:2", "2:3",
    "1:1",
]


# =============================================================================
# ASPECT RATIO DIMENSION MAPS
# =============================================================================

# Numeric values for closest-match detection
ASPECT_RATIO_VALUES = {
    "1:1": 1.0,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "2:1": 2.0,
    "1:2": 0.5,
    "19.5:9": 19.5 / 9,
    "9:19.5": 9 / 19.5,
    "20:9": 20 / 9,
    "9:20": 9 / 20,
}

# Estimated output dimensions for metadata reporting
IMAGE_ASPECT_DIMENSIONS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 756),
    "9:16": (756, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "2:1": (1448, 724),
    "1:2": (724, 1448),
    "19.5:9": (1504, 694),
    "9:19.5": (694, 1504),
    "20:9": (1520, 684),
    "9:20": (684, 1520),
}

VIDEO_ASPECT_DIMENSIONS = {
    "720p": {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (960, 720),
        "3:4": (720, 960),
        "3:2": (1080, 720),
        "2:3": (720, 1080),
        "1:1": (720, 720),
    },
    "480p": {
        "16:9": (854, 480),
        "9:16": (480, 854),
        "4:3": (640, 480),
        "3:4": (480, 640),
        "3:2": (720, 480),
        "2:3": (480, 720),
        "1:1": (480, 480),
    },
}


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """
    Configuration for exponential backoff retry on 429 rate limit errors.

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
    base_delay_ms: float = 300.0
    multiplier: float = 2.0

    def get_max_delay_ms(self, attempt: int) -> float:
        """Calculate max delay for a given attempt number (1-indexed)."""
        return self.base_delay_ms * (self.multiplier ** (attempt - 1))

    def get_jittered_delay_s(self, attempt: int) -> float:
        """Get a random delay between 0 and max_delay for the given attempt."""
        max_delay_ms = self.get_max_delay_ms(attempt)
        delay_ms = random.uniform(0, max_delay_ms)
        return delay_ms / 1000.0


DEFAULT_RETRY_CONFIG = RetryConfig()

_retry_logger = logging.getLogger(__name__ + ".retry")


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an exception is a rate limit / resource exhausted error.

    Handles both HTTP 429 errors and gRPC RESOURCE_EXHAUSTED errors
    from the xAI SDK.

    Args:
        error: The exception to check

    Returns:
        True if it's a rate limit error
    """
    error_str = str(error)
    return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str


async def retry_on_rate_limit(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> T:
    """
    Execute a sync function with exponential backoff retry on 429 errors.

    Only retries on 429 rate limit errors. All other errors are raised immediately.
    Uses jitter: delay is random between 0 and max_delay for each attempt.

    Args:
        func: Sync callable to execute (should be a zero-argument lambda or partial)
        config: Optional RetryConfig, uses DEFAULT_RETRY_CONFIG if not provided
        logger: Optional logger for retry messages

    Returns:
        The result of func() on success

    Raises:
        The original exception if max retries exceeded or non-retryable error

    Example:
        result = await retry_on_rate_limit(
            lambda: client.image.sample(model=model, prompt=prompt),
            logger=self.logger,
        )
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    log = logger or _retry_logger
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if is_rate_limit_error(e):
                last_exception = e

                if attempt < config.max_attempts:
                    delay_s = config.get_jittered_delay_s(attempt)
                    max_delay_ms = config.get_max_delay_ms(attempt)

                    log.warning(
                        f"429 rate limited on attempt {attempt}/{config.max_attempts}. "
                        f"Retrying in {delay_s*1000:.0f}ms (max: {max_delay_ms:.0f}ms)"
                    )

                    await asyncio.sleep(delay_s)
                else:
                    log.error(
                        f"429 rate limited: Max retries ({config.max_attempts}) exceeded"
                    )
                    raise
            else:
                raise

    if last_exception:
        raise last_exception

    raise RuntimeError("Unexpected state in retry_on_rate_limit")


# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

def create_xai_client() -> Client:
    """
    Create an xAI client using the XAI_API_KEY environment variable.

    Returns:
        Configured xai_sdk.Client instance

    Raises:
        RuntimeError: If XAI_API_KEY is not set
    """
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY environment variable is required")
    return Client(api_key=api_key)


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
# ASPECT RATIO UTILITIES
# =============================================================================

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


def resolve_aspect_ratio(
    aspect_ratio_value: str,
    image: Optional[File] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Resolve aspect ratio, handling 'auto' by detecting from input image.

    Args:
        aspect_ratio_value: Aspect ratio string (may be "auto")
        image: Optional input image File for auto-detection
        logger: Optional logger

    Returns:
        Resolved aspect ratio string (e.g., "16:9")
    """
    if aspect_ratio_value != "auto":
        return aspect_ratio_value

    if image and image.exists():
        try:
            from PIL import Image as PILImage
            with PILImage.open(image.path) as img:
                img_width, img_height = img.size
            resolved = find_closest_aspect_ratio(img_width, img_height)
            if logger:
                logger.info(f"Auto-detected aspect ratio: {resolved} (from {img_width}x{img_height})")
            return resolved
        except Exception as e:
            if logger:
                logger.warning(f"Failed to detect aspect ratio from image: {e}, using 1:1")
            return "1:1"
    else:
        if logger:
            logger.info("No input image for auto aspect ratio detection, using 1:1")
        return "1:1"


def get_image_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """
    Get estimated pixel dimensions for an aspect ratio.

    Args:
        aspect_ratio: Aspect ratio string

    Returns:
        Tuple of (width, height)
    """
    return IMAGE_ASPECT_DIMENSIONS.get(aspect_ratio, (1024, 1024))


def get_video_dimensions(aspect_ratio: str, resolution: str = "720p") -> tuple[int, int]:
    """
    Get pixel dimensions for a video aspect ratio and resolution.

    Args:
        aspect_ratio: Aspect ratio string
        resolution: Resolution string ("720p" or "480p")

    Returns:
        Tuple of (width, height)
    """
    res_map = VIDEO_ASPECT_DIMENSIONS.get(resolution, VIDEO_ASPECT_DIMENSIONS["720p"])
    return res_map.get(aspect_ratio, (1280, 720))


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def encode_image_base64(image: File) -> str:
    """
    Read an image file and return a base64 data URI string.

    Args:
        image: File object with path and optional content_type

    Returns:
        Data URI string (e.g., "data:image/jpeg;base64,...")

    Raises:
        RuntimeError: If the image file doesn't exist
    """
    if not image.exists():
        raise RuntimeError(f"Input image does not exist at path: {image.path}")

    with open(image.path, "rb") as f:
        image_bytes = f.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")

    content_type = image.content_type or "image/jpeg"
    return f"data:{content_type};base64,{base64_string}"


def save_image_from_response(response) -> File:
    """
    Save an xAI SDK image response to a temporary file.

    Handles both direct image bytes and URL responses.

    Args:
        response: xAI SDK image response object

    Returns:
        File object pointing to the saved image

    Raises:
        RuntimeError: If the response format is unexpected
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        if hasattr(response, 'image') and response.image:
            f.write(response.image)
        elif hasattr(response, 'url') and response.url:
            import httpx
            img_response = httpx.get(response.url)
            img_response.raise_for_status()
            f.write(img_response.content)
        else:
            raise RuntimeError(f"Unexpected response format: {response}")
        return File(path=f.name)
