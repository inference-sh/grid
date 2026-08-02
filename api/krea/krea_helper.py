import os
import asyncio
import logging
import tempfile
from typing import Optional

import httpx

BASE_URL = "https://api.krea.ai"
JOB_POLL_INTERVAL = 3.0
JOB_POLL_TIMEOUT = 600.0
TRAIN_POLL_TIMEOUT = 7200.0


class KreaAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Krea API error {status_code}: {message}")


class KreaClient:
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        self._api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def generate(self, endpoint: str, payload: dict) -> dict:
        self.logger.info(f"POST {endpoint}")
        resp = await self.client.post(endpoint, json=payload)
        if resp.status_code not in (200, 201):
            raise KreaAPIError(resp.status_code, resp.text)
        data = resp.json()
        job_id = data.get("id") or data.get("job_id") or data.get("jobId")
        if not job_id:
            if data.get("status") == "completed" or data.get("urls") or data.get("result"):
                return data
            raise KreaAPIError(-1, f"No job ID in response: {str(data)[:300]}")
        self.logger.info(f"Job created: {job_id}")
        return await self._poll_job(job_id, timeout=JOB_POLL_TIMEOUT)

    async def train(self, payload: dict) -> dict:
        self.logger.info("POST /styles/train")
        resp = await self.client.post("/styles/train", json=payload)
        if resp.status_code not in (200, 201):
            raise KreaAPIError(resp.status_code, resp.text)
        data = resp.json()
        job_id = data.get("id") or data.get("job_id") or data.get("jobId")
        if not job_id:
            return data
        self.logger.info(f"Training job created: {job_id}")
        return await self._poll_job(job_id, timeout=TRAIN_POLL_TIMEOUT)

    async def _poll_job(self, job_id: str, timeout: float) -> dict:
        elapsed = 0.0
        while elapsed < timeout:
            resp = await self.client.get(f"/jobs/{job_id}")
            if resp.status_code != 200:
                raise KreaAPIError(resp.status_code, resp.text)
            data = resp.json()
            status = data["status"]
            if status == "completed":
                self.logger.info(f"Job {job_id} completed")
                return data
            if status in ("failed", "cancelled"):
                error = (data.get("result") or {}).get("error", "Unknown error")
                raise KreaAPIError(-1, f"Job {status}: {error}")
            self.logger.info(f"Job {job_id}: {status}")
            await asyncio.sleep(JOB_POLL_INTERVAL)
            elapsed += JOB_POLL_INTERVAL
        raise KreaAPIError(-1, f"Job {job_id} timed out after {timeout}s")

    async def upload_asset(self, file_path: str) -> str:
        """Upload a file to Krea's asset storage, return the asset URL."""
        self.logger.info(f"Uploading asset: {os.path.basename(file_path)}")
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=120,
        ) as upload_client:
            with open(file_path, "rb") as f:
                resp = await upload_client.post(
                    "/assets",
                    files={"file": (os.path.basename(file_path), f)},
                )
        if resp.status_code not in (200, 201):
            raise KreaAPIError(resp.status_code, resp.text)
        data = resp.json()
        url = data.get("image_url") or data.get("url") or data.get("asset_url")
        if not url:
            raise KreaAPIError(-1, f"No URL in asset response: {str(data)[:300]}")
        self.logger.info(f"Asset uploaded: {url[:80]}...")
        return url

    async def close(self):
        await self.client.aclose()


async def download_file(
    url: str,
    suffix: str = ".png",
    logger: Optional[logging.Logger] = None,
    max_retries: int = 3,
) -> str:
    if logger:
        logger.info(f"Downloading: {url[:80]}...")
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    async with httpx.AsyncClient(timeout=180) as client:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(8192):
                            f.write(chunk)
                if logger:
                    logger.info(f"Downloaded to {tmp_path}")
                return tmp_path
            except (httpx.HTTPError, httpx.StreamError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    if logger:
                        logger.warning(f"Download attempt {attempt}/{max_retries} failed: {e}. Retry in {wait}s")
                    await asyncio.sleep(wait)
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    raise RuntimeError(f"Download failed after {max_retries} attempts: {last_error}")
