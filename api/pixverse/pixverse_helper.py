import os
import uuid
import asyncio
import logging
import tempfile

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://app-api.pixverse.ai"
POLL_INTERVAL = 5.0


def get_api_key() -> str:
    api_key = os.environ.get("PIXVERSE_KEY")
    if not api_key:
        raise RuntimeError("PIXVERSE_KEY secret is not set")
    return api_key


def get_client(timeout: float = 300) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers={"API-KEY": get_api_key()},
    )


def trace_id() -> str:
    return str(uuid.uuid4())


async def api_post(client: httpx.AsyncClient, path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    logger.info(f"POST {url}")
    resp = await client.post(url, json=payload, headers={"Ai-trace-id": trace_id()})
    data = resp.json()
    if data.get("ErrCode", 0) != 0:
        raise RuntimeError(
            f"PixVerse API error: {data.get('ErrMsg', 'Unknown')} (code={data.get('ErrCode')})"
        )
    return data.get("Resp", {})


async def api_get(client: httpx.AsyncClient, path: str) -> dict:
    url = f"{BASE_URL}{path}"
    resp = await client.get(url, headers={"Ai-trace-id": trace_id()})
    data = resp.json()
    if data.get("ErrCode", 0) != 0:
        raise RuntimeError(
            f"PixVerse API error: {data.get('ErrMsg', 'Unknown')} (code={data.get('ErrCode')})"
        )
    return data.get("Resp", {})


async def upload_image(client: httpx.AsyncClient, image_path: str) -> int:
    url = f"{BASE_URL}/openapi/v2/image/upload"
    logger.info(f"Uploading image: {image_path}")
    with open(image_path, "rb") as f:
        resp = await client.post(
            url, files={"image": f}, headers={"Ai-trace-id": trace_id()}
        )
    data = resp.json()
    if data.get("ErrCode", 0) != 0:
        raise RuntimeError(f"Image upload failed: {data.get('ErrMsg')}")
    img_id = data["Resp"]["img_id"]
    logger.info(f"Image uploaded: img_id={img_id}")
    return img_id


async def upload_video(client: httpx.AsyncClient, video_path: str) -> int:
    url = f"{BASE_URL}/openapi/v2/media/upload"
    logger.info(f"Uploading video: {video_path}")
    with open(video_path, "rb") as f:
        resp = await client.post(
            url, files={"file": f}, headers={"Ai-trace-id": trace_id()}
        )
    data = resp.json()
    if data.get("ErrCode", 0) != 0:
        raise RuntimeError(f"Video upload failed: {data.get('ErrMsg')}")
    media_id = data["Resp"]["media_id"]
    logger.info(f"Video uploaded: media_id={media_id}")
    return media_id


async def upload_audio(client: httpx.AsyncClient, audio_path: str) -> int:
    url = f"{BASE_URL}/openapi/v2/media/upload"
    logger.info(f"Uploading audio: {audio_path}")
    with open(audio_path, "rb") as f:
        resp = await client.post(
            url, files={"file": f}, headers={"Ai-trace-id": trace_id()}
        )
    data = resp.json()
    if data.get("ErrCode", 0) != 0:
        raise RuntimeError(f"Audio upload failed: {data.get('ErrMsg')}")
    media_id = data["Resp"]["media_id"]
    logger.info(f"Audio uploaded: media_id={media_id}")
    return media_id


async def poll_video(client: httpx.AsyncClient, video_id: int) -> dict:
    path = f"/openapi/v2/video/result/{video_id}"
    elapsed = 0.0
    while True:
        result = await api_get(client, path)
        status = result.get("status")
        logger.info(f"Video {video_id} status={status} ({elapsed:.0f}s)")

        if status == 1:
            return result
        elif status == 7:
            raise RuntimeError("Content moderation failed — video rejected")
        elif status == 8:
            raise RuntimeError("Video generation failed")

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL


async def download_file(url: str, suffix: str = ".mp4") -> str:
    logger.info(f"Downloading: {url[:80]}...")
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as dl:
        resp = await dl.get(url)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(resp.content)
            logger.info(f"Downloaded to {f.name}")
            return f.name
