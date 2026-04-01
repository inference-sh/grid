"""
Shared helper module for Alibaba Cloud DashScope Wan 2.7 Image API.
Provides common utilities for image generation apps using Wan-Image models.

Symlink this file into your app folder for deployment.
"""

import os
import logging
import tempfile
from typing import Optional, List, Dict, Any

import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message
import requests


# Set international API endpoint by default
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


def get_api_key() -> str:
    """Get DashScope API key from environment."""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY environment variable is required for DashScope API access."
        )
    return key


def build_message(
    prompt: str,
    reference_images: Optional[List[Any]] = None,
) -> Message:
    """
    Build a Message for the Wan ImageGeneration API.

    Args:
        prompt: Text prompt for generation/editing.
        reference_images: Optional list of File objects for image editing.

    Returns:
        Message object for the API.
    """
    content = []

    if reference_images:
        for img in reference_images:
            content.append({"image": img.uri})

    content.append({"text": prompt})

    return Message(role="user", content=content)


def generate_images(
    model: str,
    message: Message,
    num_images: int = 1,
    size: str = "2K",
    watermark: bool = False,
    thinking_mode: bool = True,
    enable_sequential: bool = False,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Call the Wan ImageGeneration API (synchronous).

    Args:
        model: Model name ("wan2.7-image" or "wan2.7-image-pro").
        message: Message object from build_message().
        num_images: Number of images to generate.
        size: Output resolution ("1K", "2K", "4K", or "WxH" pixel string).
        watermark: Whether to add "AI Generated" watermark.
        thinking_mode: Enable thinking mode for better quality (text-to-image only).
        enable_sequential: Enable image set output mode.
        seed: Random seed for reproducibility.
        logger: Optional logger.

    Returns:
        API response.

    Raises:
        RuntimeError: If API call fails.
    """
    api_key = get_api_key()

    if logger:
        logger.info(f"Calling model: {model}, n={num_images}, size={size}")

    kwargs = {
        "api_key": api_key,
        "model": model,
        "messages": [message],
        "n": num_images,
        "size": size,
        "watermark": watermark,
    }

    if enable_sequential:
        kwargs["enable_sequential"] = True
    else:
        kwargs["thinking_mode"] = thinking_mode

    if seed is not None:
        kwargs["seed"] = seed

    response = ImageGeneration.call(**kwargs)

    # Check for errors
    if hasattr(response, 'status_code') and response.status_code != 200:
        error_msg = getattr(response, 'message', 'Unknown error')
        error_code = getattr(response, 'code', 'Unknown')
        raise RuntimeError(f"DashScope API error ({error_code}): {error_msg}")

    if isinstance(response, dict):
        if response.get("code"):
            raise RuntimeError(f"DashScope API error: {response.get('message', 'Unknown error')}")

    if logger:
        logger.info("API call completed successfully")

    return response


def extract_images(
    response: Any,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Extract image URLs from Wan API response.

    Args:
        response: API response (dict or object).
        logger: Optional logger.

    Returns:
        List of image URLs.

    Raises:
        RuntimeError: If no images found in response.
    """
    image_urls = []

    if isinstance(response, dict):
        output = response.get("output", {})
        choices = output.get("choices", [])
    else:
        output = getattr(response, "output", None)
        choices = getattr(output, "choices", []) if output else []

    for choice in choices:
        if isinstance(choice, dict):
            message = choice.get("message", {})
            content_items = message.get("content", [])
        else:
            message = getattr(choice, "message", None)
            content_items = getattr(message, "content", []) if message else []

        for item in content_items:
            if isinstance(item, dict) and "image" in item:
                image_urls.append(item["image"])
            elif hasattr(item, "image"):
                image_urls.append(item.image)

    if not image_urls:
        raise RuntimeError("No images found in API response")

    if logger:
        logger.info(f"Extracted {len(image_urls)} image URL(s)")

    return image_urls


def download_image(
    url: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Download an image from URL to a temporary file."""
    if logger:
        logger.info("Downloading image...")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        file_path = tmp_file.name

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    if logger:
        logger.info(f"Image downloaded to: {file_path}")

    return file_path


def download_images(
    urls: List[str],
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Download multiple images from URLs."""
    paths = []
    for i, url in enumerate(urls):
        if logger:
            logger.info(f"Downloading image {i + 1}/{len(urls)}...")
        path = download_image(url, logger=None)
        paths.append(path)

    if logger:
        logger.info(f"Downloaded {len(paths)} image(s)")

    return paths
