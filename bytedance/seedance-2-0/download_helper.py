"""
Shared download helper for BytePlus API adapters.
Handles retry logic for CDN downloads (IncompleteRead, connection drops).
"""

import os
import time
import logging
import tempfile
from typing import Optional

import requests


def download_file(
    url: str,
    suffix: str = ".mp4",
    logger: Optional[logging.Logger] = None,
    max_retries: int = 3,
) -> str:
    """
    Download a file from URL to a temporary location.
    Retries on connection errors (e.g. IncompleteRead from CDN).

    Args:
        url: URL to download from.
        suffix: File extension for the temp file.
        logger: Optional logger for progress output.
        max_retries: Maximum number of download attempts.

    Returns:
        Path to the downloaded temporary file.
    """
    if logger:
        logger.info(f"Downloading file from: {url}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        file_path = tmp_file.name

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if logger:
                logger.info(f"File downloaded successfully to: {file_path}")
            return file_path

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                if logger:
                    logger.warning(
                        f"Download failed (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                time.sleep(wait)
            else:
                if logger:
                    logger.error(f"Download failed after {max_retries} attempts: {e}")

    # Clean up partial file on failure
    try:
        os.unlink(file_path)
    except OSError:
        pass

    raise RuntimeError(f"File download failed after {max_retries} attempts: {last_error}")


def download_video(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download a video file from URL."""
    return download_file(url, suffix=".mp4", logger=logger)


def download_image(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download an image file from URL."""
    return download_file(url, suffix=".png", logger=logger)
