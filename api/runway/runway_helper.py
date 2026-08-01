import os
import asyncio
import logging
import tempfile
from typing import Optional
from dataclasses import dataclass, field

import httpx


BASE_URL = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"

TASK_POLL_INTERVAL = 5.0
TASK_POLL_TIMEOUT = 600.0


class RunwayAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Runway API error {status_code}: {message}")


@dataclass
class RunwayTask:
    id: str
    status: str
    progress: Optional[float] = None
    output: Optional[list] = None
    failure: Optional[str] = None
    failure_code: Optional[str] = None
    created_at: Optional[str] = None


class RunwayClient:
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Runway-Version": API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def create_task(self, endpoint: str, payload: dict) -> RunwayTask:
        self.logger.info(f"POST {endpoint}")
        resp = await self.client.post(endpoint, json=payload)
        if resp.status_code not in (200, 201):
            body = resp.text
            raise RunwayAPIError(resp.status_code, body)
        data = resp.json()
        return RunwayTask(id=data["id"], status=data.get("status", "PENDING"))

    async def get_task(self, task_id: str) -> RunwayTask:
        resp = await self.client.get(f"/v1/tasks/{task_id}")
        if resp.status_code != 200:
            raise RunwayAPIError(resp.status_code, resp.text)
        data = resp.json()
        return RunwayTask(
            id=data["id"],
            status=data["status"],
            progress=data.get("progress"),
            output=data.get("output"),
            failure=data.get("failure"),
            failure_code=data.get("failureCode"),
            created_at=data.get("createdAt"),
        )

    async def poll_task(
        self,
        task_id: str,
        interval: float = TASK_POLL_INTERVAL,
        timeout: float = TASK_POLL_TIMEOUT,
    ) -> RunwayTask:
        elapsed = 0.0
        while elapsed < timeout:
            task = await self.get_task(task_id)
            if task.status == "SUCCEEDED":
                return task
            if task.status in ("FAILED", "CANCELLED"):
                msg = task.failure or task.failure_code or "Unknown error"
                raise RunwayAPIError(-1, f"Task {task.status}: {msg}")
            progress_str = f" ({task.progress}%)" if task.progress is not None else ""
            self.logger.info(f"Task {task_id}: {task.status}{progress_str}")
            await asyncio.sleep(interval)
            elapsed += interval
        raise RunwayAPIError(-1, f"Task {task_id} timed out after {timeout}s")

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
