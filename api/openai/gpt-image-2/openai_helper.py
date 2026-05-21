"""
Common utilities for OpenAI image generation models.

Shared by gpt-image-2, gpt-image-1, etc.
Provides client initialization, image downloading/saving, and size utilities.
"""

import base64
import os
import logging
import tempfile
from typing import Literal, Optional

import httpx
from openai import AsyncOpenAI
from inferencesh import File


# =============================================================================
# TYPES
# =============================================================================

QualityType = Literal["auto", "low", "medium", "high"]

OutputFormatType = Literal["png", "jpeg", "webp"]


# =============================================================================
# CLIENT
# =============================================================================

def create_openai_client() -> AsyncOpenAI:
    """
    Create an async OpenAI client using the OPENAI_KEY env var.

    Raises:
        RuntimeError: If OPENAI_KEY is not set
    """
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY environment variable is required")
    return AsyncOpenAI(api_key=api_key)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level)
    return logging.getLogger(name)


# =============================================================================
# SIZE UTILITIES
# =============================================================================

# gpt-image-2 constraints
MAX_EDGE = 3840
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_RATIO = 3


def round_to_16(value: int) -> int:
    """Round to nearest multiple of 16."""
    return round(value / 16) * 16


def validate_and_fix_dimensions(
    width: int, height: int, logger: Optional[logging.Logger] = None
) -> tuple[int, int]:
    """Validate and auto-correct dimensions for gpt-image-2.

    Rounds to nearest multiple of 16 if needed. Raises on hard constraint violations.
    Returns the (possibly adjusted) (width, height).
    """
    w, h = width, height

    # Round to nearest 16 if needed
    if w % 16 != 0 or h % 16 != 0:
        w, h = round_to_16(w), round_to_16(h)
        if logger:
            logger.info(f"Rounded dimensions to nearest 16: {width}x{height} -> {w}x{h}")

    if max(w, h) > MAX_EDGE:
        raise ValueError(f"Max edge length is {MAX_EDGE}px, got {max(w, h)}")
    pixels = w * h
    if pixels < MIN_PIXELS or pixels > MAX_PIXELS:
        raise ValueError(f"Total pixels must be {MIN_PIXELS:,}–{MAX_PIXELS:,}, got {pixels:,}")
    long_edge = max(w, h)
    short_edge = min(w, h)
    if long_edge / short_edge > MAX_RATIO:
        raise ValueError(f"Aspect ratio must not exceed {MAX_RATIO}:1, got {long_edge/short_edge:.1f}:1")

    return w, h


def make_size_string(width: int, height: int) -> str:
    """Format width/height as the size string the OpenAI API expects."""
    return f"{width}x{height}"


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def encode_image_to_base64(image: File) -> str:
    """Read an image file and return a base64-encoded string."""
    if not image.exists():
        raise RuntimeError(f"Input image does not exist at path: {image.path}")
    with open(image.path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_base64_image(b64_data: str, fmt: str = "png") -> str:
    """
    Decode base64 image data and save to a temp file.

    Returns:
        Path to the saved file.
    """
    image_bytes = base64.b64decode(b64_data)
    suffix = f".{fmt}" if fmt != "jpeg" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(image_bytes)
        return f.name


async def download_image(url: str, fmt: str = "png") -> str:
    """
    Download an image from a URL and save to a temp file.

    Returns:
        Path to the saved file.
    """
    suffix = f".{fmt}" if fmt != "jpeg" else ".jpg"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(response.content)
        return f.name
