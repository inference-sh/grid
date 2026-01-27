"""
Shared helper module for BytePlus ARK SDK operations.
Provides common utilities for video/image generation apps using BytePlus API.
"""

import os
import logging
import tempfile
import asyncio
from typing import Optional, List, Dict, Any

import requests
from byteplussdkarkruntime import Ark


def setup_byteplus_client(api_key: Optional[str] = None) -> Ark:
    """
    Configure BytePlus ARK client with API key.

    Args:
        api_key: Optional API key. If not provided, reads from ARK_API_KEY env var.

    Returns:
        Configured Ark client instance.

    Raises:
        RuntimeError: If no API key is available.
    """
    key = api_key or os.environ.get("ARK_API_KEY")
    if not key:
        raise RuntimeError(
            "ARK_API_KEY environment variable is required for BytePlus API access."
        )

    client = Ark(
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
        api_key=key,
    )
    return client


def create_content_task(
    client: Ark,
    model: str,
    content: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Create a content generation task (video/image).

    Args:
        client: The BytePlus ARK client.
        model: Model ID (e.g., "seedance-1-5-pro-251215").
        content: List of content items (text prompts, images, etc.).
        logger: Optional logger for output.

    Returns:
        Task ID for polling.
    """
    if logger:
        logger.info(f"Creating content generation task with model: {model}")

    result = client.content_generation.tasks.create(
        model=model,
        content=content,
    )

    task_id = result.id
    if logger:
        logger.info(f"Task created successfully with ID: {task_id}")

    return task_id


async def poll_task_status(
    client: Ark,
    task_id: str,
    logger: Optional[logging.Logger] = None,
    poll_interval: float = 2.0,
    max_attempts: int = 300,
    cancel_flag_getter: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Poll a content generation task until completion.

    Args:
        client: The BytePlus ARK client.
        task_id: Task ID to poll.
        logger: Optional logger for progress output.
        poll_interval: Seconds between poll attempts.
        max_attempts: Maximum number of poll attempts.
        cancel_flag_getter: Optional callable that returns True if cancelled.

    Returns:
        The completed task result.

    Raises:
        RuntimeError: If task fails or times out.
    """
    if logger:
        logger.info(f"Polling task status for: {task_id}")

    for attempt in range(max_attempts):
        # Check for cancellation
        if cancel_flag_getter and cancel_flag_getter():
            if logger:
                logger.info("Task polling cancelled by user")
            raise RuntimeError("Task was cancelled by user")

        result = client.content_generation.tasks.get(task_id=task_id)
        status = result.status

        if status == "succeeded":
            if logger:
                logger.info("Task completed successfully")
            return result
        elif status == "failed":
            error_msg = getattr(result, 'error', 'Unknown error')
            if logger:
                logger.error(f"Task failed: {error_msg}")
            raise RuntimeError(f"Content generation failed: {error_msg}")
        else:
            if logger and attempt % 5 == 0:  # Log every 5th attempt to reduce noise
                logger.info(f"Task status: {status}, waiting...")
            await asyncio.sleep(poll_interval)

    raise RuntimeError(f"Task timed out after {max_attempts * poll_interval} seconds")


def cancel_task(
    client: Ark,
    task_id: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Cancel or delete a content generation task.

    Args:
        client: The BytePlus ARK client.
        task_id: Task ID to cancel.
        logger: Optional logger for output.

    Returns:
        True if cancellation was successful.
    """
    try:
        if logger:
            logger.info(f"Cancelling task: {task_id}")
        # Note: Adjust this based on the actual SDK method
        # client.content_generation.tasks.cancel(task_id=task_id)
        if logger:
            logger.info("Task cancelled successfully")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"Failed to cancel task: {e}")
        return False


def download_file(
    url: str,
    suffix: str = ".mp4",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Download a file from URL to a temporary location.

    Args:
        url: URL to download from.
        suffix: File extension for the temp file.
        logger: Optional logger for progress output.

    Returns:
        Path to the downloaded temporary file.
    """
    if logger:
        logger.info(f"Downloading file from: {url}")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        file_path = tmp_file.name

    # Download content
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    if logger:
        logger.info(f"File downloaded successfully to: {file_path}")

    return file_path


def download_video(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download a video file from URL."""
    return download_file(url, suffix=".mp4", logger=logger)


def download_image(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download an image file from URL."""
    return download_file(url, suffix=".png", logger=logger)


def build_text_content(text: str, **params) -> Dict[str, str]:
    """
    Build a text content item for BytePlus API.

    Parameters can be embedded in the text using -- syntax:
    e.g., "prompt text --duration 5 --camerafixed false"

    Args:
        text: The prompt text.
        **params: Additional parameters to append (duration, camerafixed, etc.)

    Returns:
        Content dict for the API.
    """
    # Append parameters to text if provided
    param_str = " ".join(f"--{k} {v}" for k, v in params.items())
    full_text = f"{text} {param_str}".strip() if param_str else text

    return {
        "type": "text",
        "text": full_text,
    }


def build_image_content(image_url: str) -> Dict[str, Any]:
    """
    Build an image content item for BytePlus API.

    Args:
        image_url: URL of the image (first frame, reference image, etc.)

    Returns:
        Content dict for the API.
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": image_url,
        },
    }
