"""Tripo's own API (https://openapi.tripo3d.ai/v3), shared by the tripo apps: upload, create a task, wait for it, download.

Every call answers `{"code": 0, "data": {...}}` on success and
`{code, message, suggestion}` with a non-zero code otherwise. Tasks are polled
at `GET /tasks/{task_id}`; Tripo bills in credits ($0.01 each), reported per
task as `credits_consumed`.
"""

import asyncio
import logging
import os
import tempfile
import time
from typing import Optional

import httpx

BASE_URL = "https://openapi.tripo3d.ai/v3"
API_KEY_SECRET = "TRIPO_API_KEY"

MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
TASK_TIMEOUT = 30 * 60
POLL_INTERVAL = 3
FAILED_STATUSES = ("failed", "cancelled", "banned", "expired")

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Tripo answered with an error, or a task failed. The message carries Tripo's code and reason."""


def _error(response: httpx.Response, body: dict) -> UpstreamError:
    return UpstreamError(
        "Tripo %d, code %s: %s %s" % (response.status_code, body.get("code"), body.get("message", ""), body.get("suggestion", ""))
    )


def _body(response: httpx.Response) -> dict:
    try:
        return response.json()
    except ValueError:
        return {"message": response.text[:500]}


class TripoClient:
    def __init__(self):
        api_key = os.environ.get(API_KEY_SECRET)
        if not api_key:
            raise ValueError(f"{API_KEY_SECRET} is not set: connect the Tripo credential")
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300, connect=15),
        )

    async def request(self, method: str, path: str, **kwargs) -> dict:
        """One call returning `data`: 429 and 5xx retried, raises with Tripo's code otherwise."""
        for attempt in range(MAX_RETRIES + 1):
            response = await self.http.request(method, path, **kwargs)
            if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                logger.warning("%s %s: %d, retrying in %.0fs", method, path, response.status_code, delay)
                await asyncio.sleep(delay)
                continue
            body = _body(response)
            if response.is_error or body.get("code", 0) != 0:
                raise _error(response, body)
            return body.get("data") or {}

    async def upload(self, filename: str, data: bytes, content_type: str) -> str:
        """`POST /files`: an image or a model, referenced afterwards by the file token returned."""
        token = (await self.request("POST", "/files", files={"file": (filename, data, content_type)}))["file_token"]
        logger.info("uploaded %s (%d KB) as %s", filename, len(data) // 1024, token)
        return token

    async def create(self, path: str, body: dict) -> str:
        """Create a task. It has no idempotency key: only a 429 (rate or concurrency limit) is resent, since no task was made."""
        for attempt in range(MAX_RETRIES + 1):
            response = await self.http.post(path, json=body)
            if response.status_code == 429 and attempt < MAX_RETRIES:
                delay = float(response.headers.get("Retry-After", 10 * (attempt + 1)))
                logger.warning("create %s: 429 %s, retrying in %.0fs", path, response.text[:200], delay)
                await asyncio.sleep(delay)
                continue
            data = _body(response)
            if response.is_error or data.get("code", 0) != 0:
                raise _error(response, data)
            task_id = data["data"]["task_id"]
            logger.info("created %s task %s", path, task_id)
            return task_id

    async def wait(self, task_id: str) -> dict:
        started = time.monotonic()
        last_progress = -1
        while time.monotonic() - started < TASK_TIMEOUT:
            task = await self.request("GET", "/tasks/%s" % task_id)
            status = task.get("status")
            if status == "success":
                logger.info("task %s succeeded, %s credits", task_id, task.get("credits_consumed"))
                return task
            if status in FAILED_STATUSES:
                raise UpstreamError("Tripo task %s %s: %s %s" % (task_id, status, task.get("error_code", ""), task.get("error_message", "")))
            progress = task.get("progress", 0)
            if progress != last_progress:
                logger.info("task %s: %s %s%%", task_id, status, progress)
                last_progress = progress
            await asyncio.sleep(POLL_INTERVAL)
        raise UpstreamError("Tripo task %s did not finish in %d minutes" % (task_id, TASK_TIMEOUT // 60))

    async def run(self, path: str, body: dict) -> dict:
        """Create a task and wait for it."""
        return await self.wait(await self.create(path, body))

    async def close(self):
        await self.http.aclose()


def credits(task: dict) -> float:
    """A task's credits: decimals, up to two places."""
    return float(task.get("credits_consumed") or 0)


async def download(url: str, suffix: str) -> str:
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
    return out_path


def output_url(task: dict) -> Optional[str]:
    output = task.get("output") or {}
    return output.get("model_url") or output.get("pbr_model") or output.get("model")
