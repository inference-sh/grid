import os
import asyncio
import logging
import tempfile
from typing import Optional
from dataclasses import dataclass

import httpx


BASE_URL = "https://api.bfl.ai"

POLL_INTERVAL = 2.0

TERMINAL_STATUSES = {"Error", "Request Moderated", "Content Moderated"}


class BFLAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"BFL API error {status_code}: {message}")


@dataclass
class BFLResult:
    status: str
    polling_url: Optional[str] = None
    sample_url: Optional[str] = None
    draft_cache: Optional[str] = None


class BFLClient:
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "x-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def submit(self, endpoint: str, payload: dict) -> BFLResult:
        self.logger.info(f"POST {endpoint}")
        resp = await self.client.post(endpoint, json=payload)
        if resp.status_code not in (200, 201):
            raise BFLAPIError(resp.status_code, resp.text)
        data = resp.json()
        return BFLResult(
            status=data.get("status", "Pending"),
            polling_url=data.get("polling_url"),
        )

    async def poll(self, polling_url: str) -> BFLResult:
        resp = await self.client.get(polling_url, headers={"x-key": self.api_key})
        if resp.status_code != 200:
            raise BFLAPIError(resp.status_code, resp.text)
        data = resp.json()
        result = BFLResult(status=data["status"])
        if data.get("result"):
            result.sample_url = data["result"].get("sample")
            result.draft_cache = data["result"].get("draft_cache")
        return result

    async def submit_and_poll(
        self,
        endpoint: str,
        payload: dict,
        interval: float = POLL_INTERVAL,
    ) -> BFLResult:
        submitted = await self.submit(endpoint, payload)
        if not submitted.polling_url:
            raise BFLAPIError(-1, "No polling_url in response")

        elapsed = 0.0
        while True:
            await asyncio.sleep(interval)
            elapsed += interval
            result = await self.poll(submitted.polling_url)
            if result.status == "Ready":
                return result
            if result.status in TERMINAL_STATUSES:
                raise BFLAPIError(-1, f"Task failed: {result.status}")
            self.logger.info(f"Status: {result.status} ({elapsed:.0f}s)")

    async def close(self):
        await self.client.aclose()


async def download_file(
    url: str,
    suffix: str = ".mp4",
    logger: Optional[logging.Logger] = None,
    max_retries: int = 3,
) -> str:
    if logger:
        logger.info(f"Downloading: {url[:80]}...")
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    async with httpx.AsyncClient(timeout=300) as client:
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
