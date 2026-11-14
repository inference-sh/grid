"""
xAI video generation over the public REST API (https://api.x.ai/v1).

The same calls the xai_sdk video client makes over gRPC, as plain HTTPS
requests with the key in `Authorization: Bearer`:

    video.start(...)                 POST /v1/videos/generations
    video.start(video_url=...)       POST /v1/videos/edits
    video.extend_start(...)          POST /v1/videos/extensions
    video.get(request_id)            GET  /v1/videos/{request_id}

Endpoints and field names are xAI's REST reference
(docs.x.ai/developers/rest-api-reference/inference/videos).
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import httpx

from .xai_helper import XAIGenerationError

XAI_API_BASE = "https://api.x.ai/v1"

# The xai_sdk default for one call (27 minutes).
REQUEST_TIMEOUT_SECONDS = 1620.0

# 1 tick == 1e-10 USD (usage.cost_in_usd_ticks).
USD_PER_TICK = 1e-10

# HTTP status -> the gRPC status name the SDK raised for the same failure, so
# the helper's rate-limit retry and error classification read both alike.
_STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
}


class XAIHTTPError(RuntimeError):
    """A non-2xx answer from the xAI REST API."""

    def __init__(self, status: int, body: str):
        name = _STATUS_NAMES.get(status, "HTTP_ERROR")
        super().__init__(f"{name} (HTTP {status}): {body}")
        self.status = status
        self.body = body


def require_xai_api_key() -> str:
    """The XAI_API_KEY environment variable, or a RuntimeError when it is not set."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY environment variable is required")
    return api_key


def _request(method: str, path: str, body: Optional[dict] = None) -> dict:
    # The key is read per request and only ever goes in this header.
    headers = {"Authorization": f"Bearer {require_xai_api_key()}"}
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as http:
        response = http.request(method, f"{XAI_API_BASE}{path}", headers=headers, json=body)
    if response.status_code >= 400:
        raise XAIHTTPError(response.status_code, response.text[:2000])
    return response.json()


@dataclass
class VideoResult:
    """A finished video request: the fields the apps read from the SDK's VideoResponse."""

    respect_moderation: bool
    duration: int
    cost_usd: Optional[float]
    _url: str = ""

    @property
    def url(self) -> str:
        if not self._url:
            if not self.respect_moderation:
                raise ValueError("Video did not respect moderation rules; URL is not available.")
            raise ValueError("Video URL missing from response.")
        return self._url


def _video_result(payload: dict) -> VideoResult:
    video = payload.get("video") or {}
    ticks = (payload.get("usage") or {}).get("cost_in_usd_ticks")
    return VideoResult(
        respect_moderation=video.get("respect_moderation", True),
        duration=int(video.get("duration") or 0),
        cost_usd=ticks * USD_PER_TICK if ticks is not None else None,
        _url=video.get("url") or "",
    )


def start_generation(
    *,
    model: str,
    prompt: str,
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    image_url: Optional[str] = None,
    reference_image_urls: Optional[Sequence[str]] = None,
) -> str:
    """Start a text-, image- or reference-to-video request. Returns the request_id."""
    body: dict = {"model": model, "prompt": prompt}
    if image_url is not None:
        body["image"] = {"url": image_url}
    if duration is not None:
        body["duration"] = duration
    if aspect_ratio is not None:
        body["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        body["resolution"] = resolution
    if reference_image_urls is not None:
        body["reference_images"] = [{"url": url} for url in reference_image_urls]
    return _request("POST", "/videos/generations", body)["request_id"]


def start_edit(*, model: str, prompt: str, video_url: str) -> str:
    """Start a video edit. Returns the request_id.

    The output keeps the input video's duration, aspect ratio and resolution:
    the endpoint takes none of them.
    """
    body = {"model": model, "prompt": prompt, "video": {"url": video_url}}
    return _request("POST", "/videos/edits", body)["request_id"]


def start_extension(*, model: str, prompt: str, video_url: str, duration: Optional[int] = None) -> str:
    """Start a video extension. Returns the request_id."""
    body: dict = {"model": model, "prompt": prompt, "video": {"url": video_url}}
    if duration is not None:
        body["duration"] = duration
    return _request("POST", "/videos/extensions", body)["request_id"]


def get_video(request_id: str) -> dict:
    """The current state of a video request."""
    return _request("GET", f"/videos/{request_id}")


async def wait_for_video(
    request_id: str,
    logger: logging.Logger,
    is_cancelled: Callable[[], bool],
    interval: float = 5.0,
) -> VideoResult:
    """
    Poll a video request until it finishes.

    No timeout (the platform owns it) and a cancel stops polling within one
    interval. Raises XAIGenerationError when the job fails or expires.
    """
    waited = 0.0
    while True:
        if is_cancelled():
            raise RuntimeError(f"Cancelled while waiting for video {request_id}")
        payload = await asyncio.to_thread(get_video, request_id)
        status = payload.get("status")
        if status == "done":
            if not payload.get("video"):
                raise XAIGenerationError("xAI finished the video request but returned no response.")
            return _video_result(payload)
        if status == "expired":
            raise XAIGenerationError("xAI's video request expired before it finished.")
        if status == "failed":
            error = payload.get("error") or {}
            code = error.get("code") or "unknown"
            message = error.get("message") or "no error details"
            print(f"xAI video {request_id} failed: code={code} message={message}")
            raise XAIGenerationError(f"xAI video generation failed ({code}): {message}")
        if waited and int(waited) % 30 == 0:
            logger.info(f"Video {request_id} still pending after {waited:.0f}s")
        await asyncio.sleep(interval)
        waited += interval
