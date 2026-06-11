"""
Shared helper module for Reve API operations.
Symlink this file into your app folder for deployment.
"""

import os
import logging
import tempfile
import base64
from typing import Optional, List, Dict, Any

import requests


REVE_BASE_URL = "https://api.reve.com"


def get_api_key() -> str:
    key = os.environ.get("REVE_KEY")
    if not key:
        raise RuntimeError("REVE_KEY environment variable is required")
    return key


def get_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def image_to_base64(file_path: str) -> str:
    """Convert a local image file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_base64_image(b64_data: str, logger: Optional[logging.Logger] = None) -> str:
    """Save base64 image data to a temp file. Returns the file path."""
    image_bytes = base64.b64decode(b64_data)

    # Detect format from magic bytes
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        suffix = ".png"
    elif image_bytes[:2] == b'\xff\xd8':
        suffix = ".jpg"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        suffix = ".webp"
    else:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        path = tmp.name

    if logger:
        logger.info(f"Image saved to: {path} ({len(image_bytes)} bytes)")
    return path


def create_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    version: str = "latest",
    test_time_scaling: Optional[int] = None,
    postprocessing: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Generate an image from a text prompt."""
    if logger:
        logger.info(f"Creating image: {prompt[:80]}...")

    body: Dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "version": version,
    }
    if test_time_scaling is not None and test_time_scaling > 1:
        body["test_time_scaling"] = test_time_scaling
    if postprocessing:
        body["postprocessing"] = postprocessing

    resp = requests.post(
        f"{REVE_BASE_URL}/v1/image/create",
        headers=get_headers(),
        json=body,
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Reve create failed ({resp.status_code}): {resp.text}")

    result = resp.json()
    if logger:
        logger.info(f"Credits used: {result.get('credits_used')}, remaining: {result.get('credits_remaining')}")
    return result


def edit_image(
    edit_instruction: str,
    reference_image_b64: str,
    aspect_ratio: Optional[str] = None,
    version: str = "latest",
    test_time_scaling: Optional[int] = None,
    postprocessing: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Edit an image with a text instruction."""
    if logger:
        logger.info(f"Editing image: {edit_instruction[:80]}...")

    body: Dict[str, Any] = {
        "edit_instruction": edit_instruction,
        "reference_image": reference_image_b64,
        "version": version,
    }
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if test_time_scaling is not None and test_time_scaling > 1:
        body["test_time_scaling"] = test_time_scaling
    if postprocessing:
        body["postprocessing"] = postprocessing

    resp = requests.post(
        f"{REVE_BASE_URL}/v1/image/edit",
        headers=get_headers(),
        json=body,
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Reve edit failed ({resp.status_code}): {resp.text}")

    result = resp.json()
    if logger:
        logger.info(f"Credits used: {result.get('credits_used')}, remaining: {result.get('credits_remaining')}")
    return result


def remix_image(
    prompt: str,
    reference_images_b64: List[str],
    aspect_ratio: Optional[str] = None,
    version: str = "latest",
    test_time_scaling: Optional[int] = None,
    postprocessing: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Remix/combine images with a text prompt."""
    if logger:
        logger.info(f"Remixing {len(reference_images_b64)} image(s): {prompt[:80]}...")

    body: Dict[str, Any] = {
        "prompt": prompt,
        "reference_images": reference_images_b64,
        "version": version,
    }
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if test_time_scaling is not None and test_time_scaling > 1:
        body["test_time_scaling"] = test_time_scaling
    if postprocessing:
        body["postprocessing"] = postprocessing

    resp = requests.post(
        f"{REVE_BASE_URL}/v1/image/remix",
        headers=get_headers(),
        json=body,
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Reve remix failed ({resp.status_code}): {resp.text}")

    result = resp.json()
    if logger:
        logger.info(f"Credits used: {result.get('credits_used')}, remaining: {result.get('credits_remaining')}")
    return result
