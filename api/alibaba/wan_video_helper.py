"""
Shared helper module for Alibaba Cloud DashScope Wan 2.7 Video APIs.
Provides async task creation, polling, and download for video generation models
(t2v, i2v, r2v, videoedit).

All video APIs use asynchronous invocation: create task -> poll for result.

Symlink this file into your app folder for deployment.
"""

import os
import asyncio
import logging
import tempfile
from typing import Optional, Dict, Any

import requests


# DashScope international endpoint
BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
VIDEO_SYNTHESIS_URL = f"{BASE_URL}/services/aigc/video-generation/video-synthesis"
TASK_URL = f"{BASE_URL}/tasks"

POLL_INTERVAL = 15  # seconds between status checks
MAX_POLL_TIME = 3600  # 1 hour max wait


def get_api_key() -> str:
    """Get DashScope API key from environment."""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY environment variable is required."
        )
    return key


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }


def _task_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_api_key()}",
    }


def _create_video_task_sync(
    model: str,
    input_data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_data,
    }
    if parameters:
        payload["parameters"] = parameters

    if logger:
        logger.info(f"Creating task for model={model}")

    resp = requests.post(VIDEO_SYNTHESIS_URL, headers=_headers(), json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    output = data.get("output", {})
    task_id = output.get("task_id")
    task_status = output.get("task_status")

    if not task_id:
        code = data.get("code", output.get("code", "Unknown"))
        message = data.get("message", output.get("message", "Unknown error"))
        raise RuntimeError(f"Failed to create task ({code}): {message}")

    if logger:
        logger.info(f"Task created: {task_id} (status={task_status})")

    return task_id


def _poll_task_sync(
    task_id: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    import time

    url = f"{TASK_URL}/{task_id}"
    elapsed = 0

    while elapsed < MAX_POLL_TIME:
        resp = requests.get(url, headers=_task_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        output = data.get("output", {})
        status = output.get("task_status", "UNKNOWN")

        if logger:
            logger.info(f"Task {task_id}: {status} ({elapsed}s elapsed)")

        if status == "SUCCEEDED":
            return data

        if status in ("FAILED", "CANCELED"):
            code = output.get("code", "Unknown")
            message = output.get("message", "Unknown error")
            raise RuntimeError(f"Task failed ({code}): {message}")

        if status == "UNKNOWN":
            raise RuntimeError(f"Task {task_id} not found or expired")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise RuntimeError(f"Task {task_id} timed out after {MAX_POLL_TIME}s")


def _download_video_sync(
    url: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    if logger:
        logger.info("Downloading video...")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    if logger:
        logger.info(f"Video downloaded to: {tmp_path}")

    return tmp_path


async def create_video_task(
    model: str,
    input_data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Create an async video generation task. Returns task_id."""
    return await asyncio.to_thread(
        _create_video_task_sync, model, input_data, parameters, logger
    )


async def poll_task(
    task_id: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Poll a task until completion. Returns full response dict."""
    return await asyncio.to_thread(_poll_task_sync, task_id, logger)


async def generate_video(
    model: str,
    input_data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """End-to-end: create task + poll until done. Returns response with video_url and usage."""
    task_id = await create_video_task(model, input_data, parameters, logger)
    return await poll_task(task_id, logger)


async def download_video(
    url: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Download a video from URL to a temporary file. Returns file path."""
    return await asyncio.to_thread(_download_video_sync, url, logger)


def extract_video_url(result: Dict[str, Any]) -> str:
    """Extract video_url from a successful task result."""
    url = result.get("output", {}).get("video_url")
    if not url:
        raise RuntimeError("No video_url in task result")
    return url


def extract_usage(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract usage dict from task result."""
    return result.get("usage", {})
