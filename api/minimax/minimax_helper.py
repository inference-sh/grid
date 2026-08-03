from __future__ import annotations

import asyncio
import httpx

BASE_URL = "https://api.minimax.io/v2"


class MiniMaxError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"MiniMax API error ({status_code}): {message}")


async def create_video(client, api_key: str, payload: dict, logger) -> str:
    resp = await client.post(
        f"{BASE_URL}/video_generation",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if resp.status_code != 200:
        raise MiniMaxError(resp.status_code, _extract_error(resp))

    data = resp.json()
    logger.info(f"Create response keys: {list(data.keys())}")
    task_id = data.get("task_id") or data.get("id") or data.get("job_id")
    if not task_id:
        raise MiniMaxError(-1, f"No task ID in response: {str(data)[:300]}")
    return task_id


async def poll_video(client, api_key: str, task_id: str, logger,
                     interval: float = 3.0, timeout: float = 600.0) -> dict:
    elapsed = 0.0
    attempt = 0
    while elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
        attempt += 1

        resp = await client.get(
            f"{BASE_URL}/query/video_generation/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if resp.status_code != 200:
            logger.warning(f"Poll error ({resp.status_code}), retrying...")
            continue

        data = resp.json()
        task = data.get("task") or data
        status = task.get("status", "unknown")

        if status == "succeeded":
            url = (task.get("content") or {}).get("url") or task.get("url")
            usage = task.get("usage") or {}
            if not url:
                raise MiniMaxError(-1, f"No video URL in completed task: {str(task)[:300]}")
            return {"url": url, "usage": usage, "task": task}

        if status in ("failed", "cancelled"):
            error = (task.get("error") or {}).get("message", f"Task {status}")
            raise MiniMaxError(-1, error)

        if attempt % 10 == 0:
            logger.info(f"Status: {status} ({int(elapsed)}s elapsed)")

    raise MiniMaxError(-1, f"Video generation timed out after {int(timeout)}s")


async def download_file(client, url: str, output_path: str) -> str:
    resp = await client.get(url, follow_redirects=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    return output_path


def _extract_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        return (data.get("error") or {}).get("message") or str(data)[:300]
    except Exception:
        return resp.text[:300]
