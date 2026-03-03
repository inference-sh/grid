"""
Shared helper module for Alibaba Cloud DashScope API operations.
Provides common utilities for image generation apps using Qwen-Image models.

Symlink this file into your app folder for deployment.
"""

import os
import logging
import tempfile
from typing import Optional, List, Dict, Any

import dashscope
from dashscope import MultiModalConversation
import requests


# Set international API endpoint by default
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


def get_api_key() -> str:
    """
    Get DashScope API key from environment.

    Returns:
        The API key.

    Raises:
        RuntimeError: If no API key is available.
    """
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY environment variable is required for DashScope API access."
        )
    return key


def build_messages(
    prompt: str,
    reference_images: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build messages array for MultiModalConversation API.

    Args:
        prompt: Text prompt for generation/editing.
        reference_images: Optional list of File objects for image editing.

    Returns:
        Messages array for the API.
    """
    content = []

    # Add reference images if provided (for editing)
    if reference_images:
        for img in reference_images:
            content.append({"image": img.uri})

    # Add the text prompt
    content.append({"text": prompt})

    return [{"role": "user", "content": content}]


def generate_images(
    model: str,
    messages: List[Dict[str, Any]],
    num_images: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    watermark: bool = False,
    negative_prompt: str = "",
    prompt_extend: bool = True,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Call the Qwen-Image MultiModalConversation API.

    Args:
        model: Model name (e.g., "qwen-image-2.0", "qwen-image-2.0-pro").
        messages: Messages array from build_messages().
        num_images: Number of images to generate (1-6).
        width: Output width in pixels (512-2048).
        height: Output height in pixels (512-2048).
        watermark: Whether to add watermark.
        negative_prompt: Content to avoid in generation.
        prompt_extend: Whether to enable prompt rewriting.
        seed: Random seed for reproducibility.
        logger: Optional logger for output.

    Returns:
        API response dictionary.

    Raises:
        RuntimeError: If API call fails.
    """
    api_key = get_api_key()

    # Build size string if dimensions provided
    size = None
    if width and height:
        size = f"{width}*{height}"

    if logger:
        logger.info(f"Calling model: {model}, num_images={num_images}, size={size or 'default'}")

    # Build kwargs
    kwargs = {
        "api_key": api_key,
        "model": model,
        "messages": messages,
        "result_format": "message",
        "stream": False,
        "n": num_images,
        "watermark": watermark,
        "prompt_extend": prompt_extend,
    }

    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if size:
        kwargs["size"] = size

    if seed is not None:
        kwargs["seed"] = seed

    response = MultiModalConversation.call(**kwargs)

    # Check for errors
    if hasattr(response, 'status_code') and response.status_code != 200:
        error_msg = getattr(response, 'message', 'Unknown error')
        error_code = getattr(response, 'code', 'Unknown')
        raise RuntimeError(f"DashScope API error ({error_code}): {error_msg}")

    # Handle dict response format
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
    Extract image URLs from API response.

    Args:
        response: API response (dict or object).
        logger: Optional logger for output.

    Returns:
        List of image URLs.

    Raises:
        RuntimeError: If no images found in response.
    """
    image_urls = []

    # Handle dict response
    if isinstance(response, dict):
        output = response.get("output", {})
        choices = output.get("choices", [])
    else:
        # Handle object response
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
    """
    Download an image from URL to a temporary file.

    Args:
        url: URL to download from.
        logger: Optional logger for progress output.

    Returns:
        Path to the downloaded temporary file.
    """
    if logger:
        logger.info(f"Downloading image...")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        file_path = tmp_file.name

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    if logger:
        logger.info(f"Image downloaded to: {file_path}")

    return file_path


def download_images(
    urls: List[str],
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Download multiple images from URLs.

    Args:
        urls: List of URLs to download.
        logger: Optional logger for progress output.

    Returns:
        List of paths to downloaded temporary files.
    """
    paths = []
    for i, url in enumerate(urls):
        if logger:
            logger.info(f"Downloading image {i + 1}/{len(urls)}...")
        path = download_image(url, logger=None)
        paths.append(path)

    if logger:
        logger.info(f"Downloaded {len(paths)} image(s)")

    return paths
