"""
Shared helper module for Pruna API operations.
Provides common utilities for image/video generation apps using Pruna P-API.

Symlink this file into your app folder for deployment.
"""

import os
import logging
import tempfile
import asyncio
from typing import Optional, List, Dict, Any

import requests


PRUNA_BASE_URL = "https://api.pruna.ai/v1"


def get_api_key() -> str:
    """
    Get Pruna API key from environment.

    Returns:
        The API key.

    Raises:
        RuntimeError: If no API key is available.
    """
    key = os.environ.get("PRUNA_KEY")
    if not key:
        raise RuntimeError(
            "PRUNA_KEY environment variable is required for Pruna API access."
        )
    return key


def get_headers(model: Optional[str] = None, try_sync: bool = False) -> Dict[str, str]:
    """
    Build request headers for Pruna API.

    Args:
        model: Model name for the Model header (e.g., "p-image").
        try_sync: Whether to request synchronous mode.

    Returns:
        Headers dictionary.
    """
    headers = {
        "apikey": get_api_key(),
        "Content-Type": "application/json",
    }
    if model:
        headers["Model"] = model
    if try_sync:
        headers["Try-Sync"] = "true"
    return headers


def upload_file(
    file_path: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Upload a file to Pruna for use in image editing.

    Args:
        file_path: Path to the local file.
        logger: Optional logger for output.

    Returns:
        Upload response with file ID and URLs.

    Raises:
        RuntimeError: If upload fails.
    """
    if logger:
        logger.info(f"Uploading file: {file_path}")

    url = f"{PRUNA_BASE_URL}/files"
    headers = {"apikey": get_api_key()}

    with open(file_path, "rb") as f:
        files = {"content": f}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code != 200:
        raise RuntimeError(f"File upload failed: {response.status_code} - {response.text}")

    result = response.json()
    if logger:
        logger.info(f"File uploaded: {result.get('id')}")

    return result


def create_prediction(
    model: str,
    input_data: Dict[str, Any],
    try_sync: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Create a prediction request.

    Args:
        model: Model name (e.g., "p-image", "p-image-edit", "p-video").
        input_data: Input parameters for the model.
        try_sync: Whether to use synchronous mode (60s timeout).
        logger: Optional logger for output.

    Returns:
        Prediction response (result if sync succeeded, or task info for async).

    Raises:
        RuntimeError: If request fails.
    """
    if logger:
        logger.info(f"Creating prediction with model: {model}")

    url = f"{PRUNA_BASE_URL}/predictions"
    headers = get_headers(model=model, try_sync=try_sync)
    body = {"input": input_data}

    response = requests.post(url, headers=headers, json=body)

    if response.status_code not in (200, 201):
        raise RuntimeError(f"Prediction request failed: {response.status_code} - {response.text}")

    result = response.json()

    if logger:
        status = result.get("status", "submitted")
        logger.info(f"Prediction status: {status}")

    return result


async def poll_prediction_status(
    prediction_id: str,
    logger: Optional[logging.Logger] = None,
    poll_interval: float = 2.0,
    max_attempts: int = 300,
) -> Dict[str, Any]:
    """
    Poll a prediction until completion.

    Args:
        prediction_id: Prediction ID to poll.
        logger: Optional logger for progress output.
        poll_interval: Seconds between poll attempts.
        max_attempts: Maximum number of poll attempts.

    Returns:
        The completed prediction result.

    Raises:
        RuntimeError: If prediction fails or times out.
    """
    if logger:
        logger.info(f"Polling prediction status: {prediction_id}")

    url = f"{PRUNA_BASE_URL}/predictions/status/{prediction_id}"
    headers = {"apikey": get_api_key()}

    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"Status check failed: {response.status_code} - {response.text}")

        result = response.json()
        status = result.get("status")

        if status == "succeeded":
            if logger:
                logger.info("Prediction completed successfully")
            return result
        elif status == "failed":
            error_msg = result.get("error", result.get("message", "Unknown error"))
            if logger:
                logger.error(f"Prediction failed: {error_msg}")
            raise RuntimeError(f"Prediction failed: {error_msg}")
        else:
            if logger and attempt % 5 == 0:
                logger.info(f"Status: {status}, waiting...")
            await asyncio.sleep(poll_interval)

    raise RuntimeError(f"Prediction timed out after {max_attempts * poll_interval} seconds")


def get_generation_url(result: Dict[str, Any]) -> str:
    """
    Extract and normalize generation_url from a prediction result.

    Handles the API returning either a string or a list of strings,
    and resolves relative URLs to absolute Pruna API URLs.

    Args:
        result: Prediction result dictionary.

    Returns:
        Absolute URL string.

    Raises:
        RuntimeError: If no generation_url found.
    """
    generation_url = result.get("generation_url")
    if not generation_url:
        raise RuntimeError("No generation_url in response")
    if isinstance(generation_url, list):
        generation_url = generation_url[0]
    if generation_url.startswith("/"):
        generation_url = f"https://api.pruna.ai{generation_url}"
    return generation_url


def download_result(
    generation_url: str,
    suffix: str = ".jpg",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Download a generated result from Pruna.

    Args:
        generation_url: URL to download from.
        suffix: File extension for the temp file.
        logger: Optional logger for output.

    Returns:
        Path to the downloaded temporary file.
    """
    if logger:
        logger.info(f"Downloading result...")

    headers = {"apikey": get_api_key()}

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        file_path = tmp_file.name

    response = requests.get(generation_url, headers=headers, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    if logger:
        logger.info(f"Downloaded to: {file_path}")

    return file_path


def download_image(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download an image result."""
    return download_result(url, suffix=".jpg", logger=logger)


def download_video(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download a video result."""
    return download_result(url, suffix=".mp4", logger=logger)


async def run_prediction(
    model: str,
    input_data: Dict[str, Any],
    use_sync: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run a prediction, handling both sync and async modes.

    For sync mode, attempts synchronous request first.
    For async mode (or if sync times out), polls until complete.

    Args:
        model: Model name.
        input_data: Input parameters.
        use_sync: Whether to try sync mode first.
        logger: Optional logger.

    Returns:
        Completed prediction result with generation_url.
    """
    result = create_prediction(
        model=model,
        input_data=input_data,
        try_sync=use_sync,
        logger=logger,
    )

    # Check if sync mode returned a completed result
    if result.get("status") == "succeeded":
        return result

    # Otherwise, poll for completion
    prediction_id = result.get("id")
    if not prediction_id:
        raise RuntimeError("No prediction ID in response")

    return await poll_prediction_status(
        prediction_id=prediction_id,
        logger=logger,
    )
