import os
import base64
import tempfile
import logging
import requests
from PIL import Image

BASE_URL = "https://api.photalabs.com/v1/phota"


def get_api_key() -> str:
    key = os.environ.get("PHOTA_KEY")
    if not key:
        raise RuntimeError("PHOTA_KEY environment variable is required")
    return key


def get_headers() -> dict:
    return {
        "X-API-Key": get_api_key(),
        "Content-Type": "application/json",
    }


def phota_request(endpoint: str, payload: dict, logger: logging.Logger) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    logger.info(f"POST {url}")

    try:
        resp = requests.post(url, json=payload, headers=get_headers(), timeout=600)
    except requests.exceptions.Timeout:
        raise RuntimeError("Phota API timed out after 600s — their server may be overloaded, please retry")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Failed to connect to Phota API — their server may be down")

    if resp.status_code == 401:
        raise RuntimeError("Invalid or missing PHOTA_KEY")
    if resp.status_code == 402:
        raise RuntimeError("Insufficient Phota credit balance")
    if resp.status_code == 404:
        raise RuntimeError(_format_error(resp, "NOT_FOUND", "Resource not found"))
    if resp.status_code == 400:
        raise RuntimeError(_format_error(resp, "BAD_REQUEST", "Invalid request"))
    if resp.status_code >= 500:
        try:
            data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            raise RuntimeError(f"Phota server error ({resp.status_code}) [req={resp.headers.get('X-Request-Id', '?')}]: {resp.text or 'empty response'}")
        raise RuntimeError(f"Phota server error [req={data.get('request_id', resp.headers.get('X-Request-Id', '?'))}]: {data.get('detail', resp.text)}")

    resp.raise_for_status()
    return resp.json()


def _format_error(resp: requests.Response, default_code: str, default_detail: str) -> str:
    req_id = resp.headers.get("X-Request-Id", "?")
    try:
        data = resp.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return f"{default_code} [req={req_id}]: {resp.text or default_detail}"
    req_id = data.get("request_id", req_id)
    return f"{data.get('code', default_code)} [req={req_id}]: {data.get('detail', default_detail)}"


def save_output_images(result: dict, output_format: str, logger: logging.Logger) -> list[str]:
    """Save images from a response (bytes or urls mode) to temp files."""
    ext = "jpg" if output_format == "jpg" else "png"
    paths: list[str] = []

    urls = result.get("download_urls") or []
    images = result.get("images") or []

    if urls:
        for i, url in enumerate(urls):
            path = _tempfile_path(ext)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            logger.info(f"Downloaded image {i+1} ({len(r.content)} bytes) to {path}")
            paths.append(path)
    else:
        for i, b64 in enumerate(images):
            path = _tempfile_path(ext)
            image_bytes = base64.b64decode(b64)
            with open(path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Saved image {i+1} ({len(image_bytes)} bytes) to {path}")
            paths.append(path)

    return paths


def _tempfile_path(ext: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        return tmp.name


def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_dimensions(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def resolve_image_input(image_input) -> str:
    """Convert a File input to either a URL (if uri is http) or base64 string."""
    if hasattr(image_input, "uri") and image_input.uri and image_input.uri.startswith("http"):
        return image_input.uri
    return file_to_base64(image_input.path)
