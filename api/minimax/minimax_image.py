"""MiniMax H3 Image family (MiniMax-H3-Image, MiniMax-H3-Image-Highspeed).

Two synchronous endpoints with the same parameters:
  POST /v1/images/generations   text-to-image
  POST /v1/images/edits         1 to 16 reference images + text

The server holds the request for up to 600 s. Neither endpoint is idempotent:
a retried request generates and bills again, so only a 429 (nothing was
generated) is retried.
"""
from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from typing import List, Optional, Tuple

import httpx
from inferencesh.images import ImageLimits, fit_image

BASE_URL = "https://api.minimax.io/v1"
REQUEST_TIMEOUT = httpx.Timeout(660.0, connect=30.0)

# Reference image limits: JPG/PNG/WEBP/HEIC/HEIF, each side 128..8192 px,
# long/short side <= 4:1, request body <= 64 MiB (base64 is 4/3 of the file).
MAX_SIDE = 8192
MIN_SIDE = 128
MAX_SIDE_RATIO = 4.0
BODY_BUDGET_BYTES = 44 * 1024 * 1024  # raw image bytes across all references
PASSTHROUGH_MIMES = ("image/heic", "image/heif")

# 429 codes: 1002 is the images-per-minute limit (retry), 2056 a plan usage limit (don't).
RETRY_429_ATTEMPTS = 5


class MiniMaxImageError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"MiniMax API error ({status_code}): {message}")


def image_data_uri(path: str, max_bytes: int, filename: Optional[str] = None) -> Tuple[str, int, int]:
    """A local image as (data URI, width, height), downsized to the vendor limits if needed."""
    limits = ImageLimits(
        max_long_edge=MAX_SIDE,
        max_bytes=max_bytes,
        mime_types=("image/jpeg", "image/png", "image/webp"),
    )
    data, mime = fit_image(path, limits)
    if not mime.startswith("image/"):
        mime = mimetypes.guess_type(filename or path)[0] or mime
    if not mime.startswith("image/"):
        raise ValueError(
            f"{filename or os.path.basename(path)} is not an image MiniMax reads (JPG, PNG, WEBP, HEIC, HEIF)"
        )

    width = height = 0
    if mime not in PASSTHROUGH_MIMES:
        import io
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
        name = filename or os.path.basename(path)
        if min(width, height) < MIN_SIDE:
            raise ValueError(f"{name} is {width}x{height}; each side must be at least {MIN_SIDE} px")
        if max(width, height) / min(width, height) > MAX_SIDE_RATIO:
            raise ValueError(f"{name} is {width}x{height}; the long side may be at most 4x the short side")

    return f"data:{mime};base64,{base64.b64encode(data).decode()}", width, height


def encode_references(files: list) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Input Files as data URIs plus their (width, height), sharing the body budget."""
    if not files:
        return [], []
    per_image = BODY_BUDGET_BYTES // len(files)
    uris, dims = [], []
    for f in files:
        uri, w, h = image_data_uri(f.path, per_image, getattr(f, "filename", None))
        uris.append(uri)
        dims.append((w, h))
    return uris, dims


def build_request(
    model: str,
    prompt: str,
    image_uris: List[str],
    *,
    n: int = 1,
    size: Optional[str] = None,
    aspect_ratio: str = "auto",
    area: str = "2K",
    enhance_prompt: bool = True,
    output_format: str = "jpeg",
) -> Tuple[str, dict]:
    """(endpoint path, JSON body). Reference images select the edits endpoint."""
    body: dict = {
        "model": model,
        "prompt": prompt.strip(),
        "n": n,
        "output_format": output_format,
        "response_format": "url",
    }
    if size:
        # WIDTHxHEIGHT excludes aspect_ratio and area, even as `auto`.
        body["size"] = size
    else:
        body["aspect_ratio"] = aspect_ratio
        body["area"] = area

    if image_uris:
        # Prompt enhancement is always on for edits; the field does not exist there.
        body["images"] = [{"image_url": u} for u in image_uris]
        return "/images/edits", body

    if not enhance_prompt:
        body["disable_prompt_enhanced"] = True
        # Without enhancement nothing picks the ratio, so `auto` is refused.
        if not size and aspect_ratio == "auto":
            body["aspect_ratio"] = "1:1"
    return "/images/generations", body


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text[:500]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict) and err.get("message"):
        msg = err["message"]
        # When every image fails, the per-image reasons come with the error.
        reasons = sorted({
            (item.get("error") or {}).get("message", "")
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("error")
        } - {""})
        if reasons:
            msg = f"{msg} [{'; '.join(reasons)}]"
        if data.get("request_id"):
            msg = f"{msg} (request {data['request_id']})"
        return msg
    return str(data)[:500]


async def post_images(client: httpx.AsyncClient, api_key: str, path: str, body: dict, logger) -> dict:
    """POST and return the parsed 200 body. Retries only the per-minute 429."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(1, RETRY_429_ATTEMPTS + 1):
        resp = await client.post(f"{BASE_URL}{path}", headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        message = _error_message(resp)
        if resp.status_code == 429 and "(2056)" not in message and attempt < RETRY_429_ATTEMPTS:
            try:
                wait = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                wait = 15.0 * attempt
            wait = min(max(wait, 1.0), 60.0)
            logger.warning(f"Rate limited ({message}); retry {attempt} in {wait:.0f}s")
            await asyncio.sleep(wait)
            continue
        raise MiniMaxImageError(resp.status_code, message)
    raise MiniMaxImageError(429, "rate limited")  # unreachable


async def download(client: httpx.AsyncClient, url: str, output_path: str) -> str:
    for attempt in range(3):
        resp = await client.get(url, follow_redirects=True, timeout=120.0)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return output_path
        if resp.status_code < 500 and resp.status_code != 429:
            break
        await asyncio.sleep(2.0 * (attempt + 1))
    raise MiniMaxImageError(resp.status_code, f"could not download the generated image: {resp.text[:200]}")


def usage_extra(usage: dict) -> dict:
    """Token counts from the response `usage`, flattened for pricing."""
    in_details = usage.get("input_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or {}
    return {
        "generated_images": int(usage.get("generated_images") or 0),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "input_text_tokens": int(in_details.get("text_tokens") or 0),
        "input_image_tokens": int(in_details.get("image_tokens") or 0),
        "output_image_tokens": int(out_details.get("image_tokens") or usage.get("output_tokens") or 0),
    }
