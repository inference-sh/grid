"""
Shared helper for Bria API v2 image/video editing endpoints.
Symlink this file into each app folder for deployment.

Handles authentication, async polling, and result downloading.
"""

import os
import asyncio
import logging
import tempfile
import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://engine.prod.bria-api.com/v2/image/edit"
VIDEO_EDIT_URL = "https://engine.prod.bria-api.com/v2/video/edit"
VIDEO_URL = "https://engine.prod.bria-api.com/v2/video"
POLL_INTERVAL = 1.5


def get_client(timeout: float = 120) -> httpx.AsyncClient:
    api_key = os.environ.get("BRIA_KEY")
    if not api_key:
        raise RuntimeError("BRIA_KEY secret is not set")
    return httpx.AsyncClient(
        timeout=timeout,
        headers={
            "api_token": api_key,
            "Content-Type": "application/json",
        },
    )


def _raise_for_status(resp: httpx.Response) -> None:
    """Like resp.raise_for_status(), but 4xx errors include the response body (e.g. 422 field details)."""
    if 400 <= resp.status_code < 500:
        try:
            body = resp.json()
            detail = body.get("error", body) if isinstance(body, dict) else body
            if isinstance(detail, dict):
                detail = detail.get("detail") or detail.get("message") or detail
        except Exception:
            detail = resp.text[:500]
        raise RuntimeError(f"Bria API error {resp.status_code} for {resp.request.url.path}: {detail}")
    resp.raise_for_status()


async def call_endpoint(client: httpx.AsyncClient, endpoint: str, payload: dict, *, base_url: str | None = None) -> dict:
    """Call a Bria v2 endpoint with async polling."""
    url = f"{base_url or BASE_URL}/{endpoint}"
    payload["sync"] = False

    logger.info(f"POST {url}")
    resp = await client.post(url, json=payload)
    _raise_for_status(resp)
    data = resp.json()

    request_id = data.get("request_id")
    status_url = data.get("status_url")

    if not status_url:
        # sync response came back directly
        return data

    logger.info(f"Polling request {request_id}")
    elapsed = 0.0
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        poll_resp = await client.get(status_url)
        _raise_for_status(poll_resp)
        poll_data = poll_resp.json()
        status = poll_data.get("status", "")

        if status == "COMPLETED":
            logger.info(f"Request {request_id} completed in {elapsed:.1f}s")
            return poll_data
        elif status in ("ERROR", "UNKNOWN"):
            error = poll_data.get("error", {})
            raise RuntimeError(
                f"Bria request {request_id} failed: {error.get('message', status)}"
            )


async def download_image(client: httpx.AsyncClient, image_url: str, suffix: str = ".png") -> str:
    """Download result image to a temp file, return the path."""
    resp = await client.get(image_url)
    _raise_for_status(resp)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(resp.content)
        return f.name


async def download_video(client: httpx.AsyncClient, video_url: str, suffix: str = ".mp4") -> str:
    """Download result video to a temp file, return the path."""
    resp = await client.get(video_url)
    _raise_for_status(resp)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(resp.content)
        return f.name
