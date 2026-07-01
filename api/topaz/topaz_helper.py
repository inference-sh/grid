"""
Shared helper for Topaz Labs API apps.
Handles the multi-step video processing workflow:
  1. Create request (POST /video/)
  2. Accept & get upload URL (PATCH /video/{id}/accept)
  3. Upload to S3 (PUT)
  4. Complete upload (PATCH /video/{id}/complete-upload)
  5. Poll status (GET /video/{id}/status)
  6. Download result
"""

import os
import logging
import time
import tempfile
import requests
from typing import Optional


BASE_URL = "https://api.topazlabs.com/video"


def get_api_key() -> str:
    key = os.environ.get("TOPAZ_KEY")
    if not key:
        raise RuntimeError("TOPAZ_KEY environment variable is required.")
    return key


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


def _headers(api_key: str, content_type: str = "application/json") -> dict:
    h = {
        "X-API-Key": api_key,
        "Accept": "application/json",
    }
    if content_type:
        h["Content-Type"] = content_type
    return h


def create_request(
    api_key: str,
    source: dict,
    output: dict,
    filters: list,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Step 1: Create a video processing request."""
    log = logger or logging.getLogger(__name__)

    payload = {
        "source": source,
        "output": output,
        "filters": filters,
    }

    log.info(f"Creating video request: {filters}")
    resp = requests.post(
        f"{BASE_URL}/",
        json=payload,
        headers=_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    log.info(f"Request created: {data.get('id', 'unknown')}")
    return data


def accept_request(
    api_key: str,
    request_id: str,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Step 2: Accept the request and get S3 upload URL."""
    log = logger or logging.getLogger(__name__)

    log.info(f"Accepting request {request_id}")
    resp = requests.patch(
        f"{BASE_URL}/{request_id}/accept",
        headers=_headers(api_key, content_type=None),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    log.info("Got upload URL")
    return data


def upload_to_s3(
    upload_url: str,
    file_path: str,
    content_type: str = "video/mp4",
    logger: Optional[logging.Logger] = None,
) -> str:
    """Step 3: Upload the video file to S3. Returns the eTag."""
    log = logger or logging.getLogger(__name__)

    file_size = os.path.getsize(file_path)
    log.info(f"Uploading {file_size} bytes to S3")

    with open(file_path, "rb") as f:
        resp = requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": content_type},
            timeout=600,
        )
    resp.raise_for_status()

    etag = resp.headers.get("ETag", "").strip('"')
    log.info(f"Upload complete, eTag: {etag}")
    return etag


def complete_upload(
    api_key: str,
    request_id: str,
    etag: str,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Step 4: Mark the upload as complete."""
    log = logger or logging.getLogger(__name__)

    payload = {
        "uploadResults": [{"partNum": 1, "eTag": etag}]
    }

    log.info(f"Completing upload for {request_id}")
    resp = requests.patch(
        f"{BASE_URL}/{request_id}/complete-upload",
        json=payload,
        headers=_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    log.info("Upload completed, queued for processing")
    return data


def poll_status(
    api_key: str,
    request_id: str,
    poll_interval: float = 5.0,
    max_wait: float = 1800.0,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Step 5: Poll until processing is complete."""
    log = logger or logging.getLogger(__name__)

    elapsed = 0.0
    while elapsed < max_wait:
        resp = requests.get(
            f"{BASE_URL}/{request_id}/status",
            headers=_headers(api_key, content_type=None),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "unknown")
        progress = data.get("progress", 0)
        log.info(f"Status: {status}, progress: {progress}%, elapsed: {elapsed:.0f}s")

        if status in ("completed", "complete", "done"):
            return data
        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"Processing failed: {data}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise RuntimeError(f"Processing timed out after {max_wait}s")


def download_result(
    download_url: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Step 6: Download the processed video."""
    log = logger or logging.getLogger(__name__)

    log.info(f"Downloading result")
    resp = requests.get(download_url, stream=True, timeout=600)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        output_path = tmp.name

    file_size = os.path.getsize(output_path)
    log.info(f"Downloaded {file_size} bytes to {output_path}")
    return output_path


def get_video_info(file_path: str) -> dict:
    """Get basic video info using ffprobe if available, otherwise return defaults."""
    import subprocess
    import json as json_mod

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            probe = json_mod.loads(result.stdout)
            video_stream = next(
                (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )
            fmt = probe.get("format", {})
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "duration": float(fmt.get("duration", 0)),
                "frame_rate": eval(video_stream.get("r_frame_rate", "30/1")),
                "frame_count": int(video_stream.get("nb_frames", 0)),
                "size": int(fmt.get("size", 0)),
                "container": os.path.splitext(file_path)[1].lstrip(".") or "mp4",
            }
    except Exception:
        pass

    # Fallback: just return file size
    return {
        "width": 0,
        "height": 0,
        "duration": 0,
        "frame_rate": 30,
        "frame_count": 0,
        "size": os.path.getsize(file_path),
        "container": os.path.splitext(file_path)[1].lstrip(".") or "mp4",
    }
