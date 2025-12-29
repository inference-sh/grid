"""
Shared helper module for fal.ai client operations.
Provides common utilities for video/image generation apps using fal.ai API.
"""

import os
import logging
import tempfile
from typing import Optional, Callable

import fal_client
import requests


def setup_fal_client(api_key: Optional[str] = None) -> str:
    """
    Configure fal.ai client with API key.
    
    Args:
        api_key: Optional API key. If not provided, reads from FAL_KEY env var.
        
    Returns:
        The API key that was set.
        
    Raises:
        RuntimeError: If no API key is available.
    """
    key = api_key or os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError(
            "FAL_KEY environment variable is required for fal.ai API access."
        )
    fal_client.api_key = key
    return key


def create_progress_callback(logger: logging.Logger) -> Callable:
    """
    Create a progress callback function for fal.ai subscribe.
    
    Args:
        logger: Logger instance for output.
        
    Returns:
        Callback function for on_queue_update.
    """
    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                logger.info(f"fal.ai: {log['message']}")
    return on_queue_update


def run_fal_model(
    model_id: str,
    arguments: dict,
    logger: logging.Logger,
    with_logs: bool = True,
) -> dict:
    """
    Run a fal.ai model with progress logging.
    
    Args:
        model_id: The fal.ai model endpoint ID.
        arguments: Request arguments for the model.
        logger: Logger instance for progress output.
        with_logs: Whether to enable log streaming.
        
    Returns:
        The model result dictionary.
    """
    logger.info(f"Sending request to fal.ai model: {model_id}")
    
    result = fal_client.subscribe(
        model_id,
        arguments=arguments,
        with_logs=with_logs,
        on_queue_update=create_progress_callback(logger),
    )
    
    logger.info("Model inference completed successfully")
    return result


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
