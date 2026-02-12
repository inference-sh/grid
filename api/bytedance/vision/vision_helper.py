"""
Shared helper module for BytePlus Vision API SDK operations.
Provides common utilities for OmniHuman avatar video generation apps.

Note: This uses a DIFFERENT API and SDK than the ARK API (seedream/seedance).
- Endpoint: cv.byteplusapi.com
- Auth: HMAC-SHA256 with Access Key + Secret Key
- SDK: byteplus_sdk.visual.VisualService
"""

import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any

from byteplus_sdk.visual.VisualService import VisualService

from .download_helper import download_file as _download_file


def setup_vision_service(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> VisualService:
    """
    Configure BytePlus Vision Service with Access Key and Secret Key.

    Args:
        access_key: Optional AK. If not provided, reads from BYTEPLUS_ACCESS_KEY env var.
        secret_key: Optional SK. If not provided, reads from BYTEPLUS_SECRET_KEY env var.

    Returns:
        Configured VisualService instance.

    Raises:
        RuntimeError: If credentials are not available.
    """
    ak = access_key or os.environ.get("BYTEPLUS_ACCESS_KEY")
    sk = secret_key or os.environ.get("BYTEPLUS_SECRET_KEY")

    if not ak:
        raise RuntimeError(
            "BYTEPLUS_ACCESS_KEY environment variable is required for Vision API access."
        )
    if not sk:
        raise RuntimeError(
            "BYTEPLUS_SECRET_KEY environment variable is required for Vision API access."
        )

    visual_service = VisualService()
    visual_service.set_ak(ak)
    visual_service.set_sk(sk)

    return visual_service


def detect_subjects(
    visual_service: VisualService,
    image_url: str,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Detect subjects (people) in an image and return mask URLs.

    Args:
        visual_service: The BytePlus Visual Service instance.
        image_url: URL of the portrait image.
        logger: Optional logger for output.

    Returns:
        List of mask URLs, sorted by area (largest first).

    Raises:
        RuntimeError: If detection fails or no subjects found.
    """
    if logger:
        logger.info(f"Detecting subjects in image: {image_url}")

    form = {
        "req_key": "realman_avatar_object_detection_cv",
        "image_url": image_url,
    }

    resp = visual_service.cv_process(form)

    if resp.get("code") != 10000:
        error_msg = resp.get("message", "Unknown error")
        if logger:
            logger.error(f"Subject detection failed: {error_msg}")
        raise RuntimeError(f"Subject detection failed: {error_msg}")

    # Parse nested response
    data = resp.get("data", {})
    resp_data_str = data.get("resp_data")
    if not resp_data_str:
        raise RuntimeError("Missing resp_data in subject detection response")

    inner_data = json.loads(resp_data_str)

    if inner_data.get("code") != 0:
        raise RuntimeError(f"Subject detection inner error: {inner_data}")

    status = inner_data.get("status", 0)
    if status == 0:
        if logger:
            logger.warning("No subjects detected in image")
        return []

    object_detection = inner_data.get("object_detection_result", {})
    mask = object_detection.get("mask", {})
    mask_urls = mask.get("url", [])

    if logger:
        logger.info(f"Detected {len(mask_urls)} subject(s)")

    return mask_urls


def submit_video_task(
    visual_service: VisualService,
    req_key: str,
    image_url: str,
    audio_url: str,
    mask_url: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Submit an OmniHuman video generation task.

    Args:
        visual_service: The BytePlus Visual Service instance.
        req_key: The request key for the model (e.g., "realman_avatar_picture_omni15_cv").
        image_url: URL of the portrait image.
        audio_url: URL of the audio file.
        mask_url: Optional mask URL for specifying which subject to drive.
        logger: Optional logger for output.

    Returns:
        Task ID for polling.

    Raises:
        RuntimeError: If task submission fails.
    """
    if logger:
        logger.info(f"Submitting video generation task with req_key: {req_key}")

    form = {
        "req_key": req_key,
        "image_url": image_url,
        "audio_url": audio_url,
    }

    if mask_url:
        form["mask_url"] = mask_url

    resp = visual_service.cv_submit_task(form)

    if resp.get("code") != 10000:
        error_msg = resp.get("message", "Unknown error")
        if logger:
            logger.error(f"Task submission failed: {error_msg}")
        raise RuntimeError(f"Task submission failed: {error_msg}")

    data = resp.get("data", {})
    task_id = data.get("task_id")

    if not task_id:
        raise RuntimeError(f"No task_id in response: {resp}")

    if logger:
        logger.info(f"Task submitted successfully with ID: {task_id}")

    return task_id


async def poll_video_task(
    visual_service: VisualService,
    req_key: str,
    task_id: str,
    logger: Optional[logging.Logger] = None,
    poll_interval: float = 3.0,
    max_attempts: int = 200,
    cancel_flag_getter: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Poll a video generation task until completion.

    Args:
        visual_service: The BytePlus Visual Service instance.
        req_key: The request key (must match submission).
        task_id: Task ID to poll.
        logger: Optional logger for progress output.
        poll_interval: Seconds between poll attempts.
        max_attempts: Maximum number of poll attempts.
        cancel_flag_getter: Optional callable that returns True if cancelled.

    Returns:
        Dict with video_url and other response data (comfyui_cost, etc.)

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

        form = {
            "req_key": req_key,
            "task_id": task_id,
        }

        resp = visual_service.cv_get_result(form)

        code = resp.get("code")
        data = resp.get("data", {})

        # Check for completion
        if code == 10000:
            status = data.get("status")

            # Parse response to get video URL and other data
            resp_data_str = data.get("resp_data")
            if resp_data_str and status == "done":
                inner_data = json.loads(resp_data_str)
                video_url = inner_data.get("video_url")
                if video_url:
                    if logger:
                        logger.info("Task completed successfully")
                        logger.info(f"API response data: {inner_data}")
                    return {
                        "video_url": video_url,
                        "comfyui_cost": inner_data.get("comfyui_cost"),
                        "progress": inner_data.get("progress"),
                        "vid": inner_data.get("vid"),
                        "received_at": inner_data.get("received_at"),
                        "processed_at": inner_data.get("processed_at"),
                        "finished_at": inner_data.get("finished_at"),
                    }

        # Check for failure
        if code == 50500 or code == 50501:
            error_msg = resp.get("message", "Internal error")
            if logger:
                logger.error(f"Task failed: {error_msg}")
            raise RuntimeError(f"Video generation failed: {error_msg}")

        # Still processing
        if logger and attempt % 5 == 0:
            status = data.get("status", "unknown")
            logger.info(f"Task status: {status} (attempt {attempt + 1})...")

        await asyncio.sleep(poll_interval)

    raise RuntimeError(f"Task timed out after {max_attempts * poll_interval} seconds")


def download_video(url: str, logger: Optional[logging.Logger] = None) -> str:
    """Download a video file. Delegates to shared byteplus_helper.download_file."""
    return _download_file(url, suffix=".mp4", logger=logger)


def get_video_duration(video_path: str, logger: Optional[logging.Logger] = None) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to the video file.
        logger: Optional logger for output.

    Returns:
        Duration in seconds.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            if logger:
                logger.info(f"Video duration: {duration:.2f} seconds")
            return duration

    except Exception as e:
        if logger:
            logger.warning(f"Could not get video duration via ffprobe: {e}")

    return 0.0
