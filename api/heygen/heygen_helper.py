"""
Shared helper for HeyGen API v3 endpoints.
Symlink this file into each app folder for deployment.

Handles authentication, async polling, and result downloading.
"""

import os
import asyncio
import logging
import tempfile
import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.heygen.com"
POLL_INTERVAL = 3.0
MAX_POLL_TIME = 600


def get_api_key() -> str:
    api_key = os.environ.get("HEYGEN_KEY")
    if not api_key:
        raise RuntimeError("HEYGEN_KEY secret is not set")
    return api_key


def get_client(timeout: float = 120) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers={
            "x-api-key": get_api_key(),
            "Content-Type": "application/json",
        },
    )


async def post_endpoint(client: httpx.AsyncClient, path: str, payload: dict) -> dict:
    """POST to a HeyGen v3 endpoint and return the response data."""
    url = f"{BASE_URL}{path}"
    logger.info(f"POST {url}")
    resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        logger.error(f"HeyGen API {resp.status_code}: {resp.text}")
        try:
            err = resp.json().get("error", {})
            msg = err.get("message", resp.text)
            raise RuntimeError(f"HeyGen API error ({resp.status_code}): {msg}")
        except (ValueError, AttributeError):
            resp.raise_for_status()
    data = resp.json()
    return data.get("data", data)


async def get_endpoint(client: httpx.AsyncClient, path: str) -> dict:
    """GET a HeyGen v3 endpoint and return the response data."""
    url = f"{BASE_URL}{path}"
    resp = await client.get(url)
    if resp.status_code >= 400:
        logger.error(f"HeyGen API {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data)


async def poll_video(client: httpx.AsyncClient, video_id: str) -> dict:
    """Poll GET /v3/videos/{video_id} until completed or failed."""
    path = f"/v3/videos/{video_id}"
    elapsed = 0.0
    while elapsed < MAX_POLL_TIME:
        data = await get_endpoint(client, path)
        status = data.get("status", "")
        logger.info(f"Video {video_id} status: {status} ({elapsed:.0f}s)")

        if status == "completed":
            return data
        elif status == "failed":
            msg = data.get("failure_message", "Unknown error")
            code = data.get("failure_code", "")
            raise RuntimeError(f"Video generation failed [{code}]: {msg}")

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Video {video_id} timed out after {MAX_POLL_TIME}s")


async def poll_lipsync(client: httpx.AsyncClient, lipsync_id: str) -> dict:
    """Poll GET /v3/lipsyncs/{lipsync_id} until completed or failed."""
    path = f"/v3/lipsyncs/{lipsync_id}"
    elapsed = 0.0
    while elapsed < MAX_POLL_TIME:
        data = await get_endpoint(client, path)
        status = data.get("status", "")
        logger.info(f"Lipsync {lipsync_id} status: {status} ({elapsed:.0f}s)")

        if status == "completed":
            return data
        elif status == "failed":
            msg = data.get("failure_message", "Unknown error")
            raise RuntimeError(f"Lipsync failed: {msg}")

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Lipsync {lipsync_id} timed out after {MAX_POLL_TIME}s")


async def poll_translation(client: httpx.AsyncClient, translation_id: str) -> dict:
    """Poll GET /v3/video-translations/{id} until completed or failed."""
    path = f"/v3/video-translations/{translation_id}"
    elapsed = 0.0
    while elapsed < MAX_POLL_TIME:
        data = await get_endpoint(client, path)
        status = data.get("status", "")
        logger.info(f"Translation {translation_id} status: {status} ({elapsed:.0f}s)")

        if status == "completed":
            return data
        elif status == "failed":
            msg = data.get("failure_message", "Unknown error")
            raise RuntimeError(f"Translation failed: {msg}")

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Translation {translation_id} timed out after {MAX_POLL_TIME}s")


async def download_file(url: str, suffix: str = ".mp4") -> str:
    """Download a file from URL to a temp path."""
    async with httpx.AsyncClient(timeout=300) as dl:
        resp = await dl.get(url)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(resp.content)
            return f.name


async def list_avatars(client: httpx.AsyncClient, limit: int = 20) -> list:
    """List available avatar looks."""
    data = await get_endpoint(client, f"/v3/avatars/looks?limit={limit}")
    if isinstance(data, list):
        return data
    return data.get("avatar_looks", data.get("items", []))


async def list_voices(client: httpx.AsyncClient, limit: int = 20, engine: str = None) -> list:
    """List available voices, optionally filtered by engine."""
    url = f"/v3/voices?limit={limit}"
    if engine:
        url += f"&engine={engine}"
    data = await get_endpoint(client, url)
    if isinstance(data, list):
        return data
    return data.get("voices", data.get("items", []))


def build_asset_ref(file_obj) -> dict:
    """Build a HeyGen asset reference from an inferencesh File object.

    Uses the public URI if available (url type), otherwise raises.
    """
    if file_obj.uri and file_obj.uri.startswith("http"):
        return {"type": "url", "url": file_obj.uri}
    raise RuntimeError(
        "HeyGen requires a publicly accessible URL for file inputs. "
        "Upload the file first or provide a URL."
    )
