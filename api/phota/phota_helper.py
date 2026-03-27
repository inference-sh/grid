import os
import base64
import struct
import tempfile
import logging
import requests

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

    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)

    if resp.status_code == 401:
        raise RuntimeError("Invalid or missing PHOTA_KEY")
    if resp.status_code == 402:
        raise RuntimeError("Insufficient Phota credit balance")
    if resp.status_code == 404:
        data = resp.json()
        raise RuntimeError(f"{data.get('code', 'NOT_FOUND')}: {data.get('detail', 'Resource not found')}")
    if resp.status_code == 400:
        data = resp.json()
        raise RuntimeError(f"{data.get('code', 'BAD_REQUEST')}: {data.get('detail', 'Invalid request')}")
    if resp.status_code >= 500:
        data = resp.json()
        raise RuntimeError(f"Phota server error: {data.get('detail', resp.text)}")

    resp.raise_for_status()
    return resp.json()


def save_base64_images(b64_images: list[str], logger: logging.Logger) -> list[str]:
    paths = []
    for i, b64 in enumerate(b64_images):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        image_bytes = base64.b64decode(b64)
        with open(path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved image {i+1} ({len(image_bytes)} bytes) to {path}")
        paths.append(path)
    return paths


def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_png_dimensions(path: str) -> tuple[int, int]:
    """Read width and height from a PNG file header."""
    with open(path, "rb") as f:
        f.read(8)  # skip PNG signature
        f.read(4)  # skip IHDR chunk length
        f.read(4)  # skip IHDR chunk type
        width = struct.unpack(">I", f.read(4))[0]
        height = struct.unpack(">I", f.read(4))[0]
    return width, height


def resolve_image_input(image_input) -> str:
    """Convert a File input to either a URL (if uri is http) or base64 string."""
    if hasattr(image_input, "uri") and image_input.uri and image_input.uri.startswith("http"):
        return image_input.uri
    return file_to_base64(image_input.path)
