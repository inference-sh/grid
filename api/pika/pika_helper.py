"""Shared client for the Pika API (api.dev.pika.art).

Used by every pika/* app via a symlink. Covers: presigned uploads for local
input files, async job submit + poll, result download and media probing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import subprocess
from typing import Optional

import httpx

BASE_URL = "https://api.dev.pika.art"

# Launch prices (until 2025-11-15) are half of these. Keep the regular table
# here so pricing.md and the CEL expressions can be checked against it.
GENERATION_USD_PER_SECOND = {
    "480p": 0.02, "768p": 0.04, "1080p": 0.07, "2k": 0.08, "4k": 0.16,
}
REFERENCE_USD_PER_SECOND = {
    "480p": 0.06, "768p": 0.12, "1080p": 0.15, "2k": 0.16, "4k": 0.20,
}
STEMS_USD_PER_SECOND = 0.001

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class PikaError(Exception):
    def __init__(self, message: str, status_code: int = 0, code: str = ""):
        self.status_code = status_code
        self.code = code
        prefix = f"Pika API error"
        if status_code:
            prefix += f" ({status_code})"
        if code:
            prefix += f" [{code}]"
        super().__init__(f"{prefix}: {message}")


def get_api_key() -> str:
    key = os.environ.get("PIKA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PIKA_API_KEY is not set. Connect the Pika credential on the platform "
            "(belt credentials connect pika --as platform)."
        )
    return key


def auth_headers(api_key: str) -> dict:
    return {"X-API-Key": api_key}


def guess_content_type(file, default: str) -> str:
    ct = (getattr(file, "content_type", None) or "").strip()
    if ct and ct != "application/octet-stream":
        return ct
    for attr in ("filename", "path", "uri"):
        name = getattr(file, attr, None)
        if name:
            guessed, _ = mimetypes.guess_type(str(name).split("?")[0])
            if guessed:
                return guessed
    return default


async def upload_file(client: httpx.AsyncClient, api_key: str, file, default_type: str,
                      logger: logging.Logger) -> str:
    """Return an https URL Pika can read for an input File.

    Uploads the platform's local copy through Pika's presigned upload flow so
    private or robots-blocked source URLs still work. Falls back to the
    original URL when there is no local copy.
    """
    path = getattr(file, "path", None)
    if not path or not os.path.exists(path):
        uri = getattr(file, "uri", None) or ""
        if not uri.startswith(("http://", "https://")):
            raise PikaError("input file has no local copy and no http(s) URL")
        logger.info("No local copy, passing source URL to Pika")
        return uri

    size = os.path.getsize(path)
    content_type = guess_content_type(file, default_type)
    logger.info(f"Uploading {os.path.basename(path)} ({size} bytes, {content_type}) to Pika")

    resp = await _request_with_retry(
        client, "POST", f"{BASE_URL}/v1/media/uploads",
        headers={**auth_headers(api_key), "Content-Type": "application/json"},
        json={"content_type": content_type, "size_bytes": size},
        logger=logger,
    )
    data = resp.json()
    url = data.get("url")
    upload_url = data.get("upload_url")
    if not url:
        raise PikaError(f"upload response has no url: {str(data)[:300]}")
    if not upload_url:
        # Legacy response: the file is already addressable by url.
        return url

    headers = dict(data.get("headers") or {})
    headers.setdefault("Content-Type", content_type)
    headers.setdefault("Content-Length", str(size))
    with open(path, "rb") as f:
        body = f.read()
    put = await client.request(data.get("method") or "PUT", upload_url, content=body, headers=headers)
    if put.status_code >= 300:
        raise PikaError(f"upload PUT failed: {put.text[:300]}", put.status_code)
    return url


async def submit_job(client: httpx.AsyncClient, api_key: str, path: str, payload: dict,
                     logger: logging.Logger) -> str:
    """POST a media operation and return the job id."""
    url = f"{BASE_URL}{path}"
    logger.info(f"POST {path} keys={sorted(payload.keys())}")
    resp = await _request_with_retry(
        client, "POST", url,
        headers={**auth_headers(api_key), "Content-Type": "application/json"},
        json=payload, logger=logger,
        # A submit creates a job row even on 429/503 (failed envelope), so only
        # retry when the server explicitly asks us to.
        retry_only_with_retry_after=True,
    )
    job = resp.json()
    _raise_if_failed(job)
    job_id = job.get("id")
    if not job_id:
        raise PikaError(f"submit response has no job id: {str(job)[:300]}")
    logger.info(f"Job {job_id} status={job.get('status')}")
    return job_id


async def poll_job(client: httpx.AsyncClient, api_key: str, job_id: str, logger: logging.Logger,
                   interval: float = 3.0, timeout: float = 3600.0) -> dict:
    """Poll until the job is terminal. Returns the completed job object."""
    elapsed = 0.0
    last_status = None
    last_log = 0.0
    while elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
        try:
            resp = await client.get(f"{BASE_URL}/v1/media/jobs/{job_id}", headers=auth_headers(api_key))
        except httpx.HTTPError as e:
            logger.warning(f"Poll network error, retrying: {e}")
            continue
        if resp.status_code in RETRYABLE_STATUSES:
            logger.warning(f"Poll HTTP {resp.status_code}, retrying")
            continue
        if resp.status_code >= 400:
            raise PikaError(_extract_message(resp), resp.status_code)

        job = resp.json()
        status = job.get("status")
        if status != last_status or elapsed - last_log >= 30:
            logger.info(f"Job {job_id}: {status} ({int(elapsed)}s)")
            last_status, last_log = status, elapsed

        if status == "completed":
            job = await _settle_billing(client, api_key, job_id, job, logger)
            _log_terminal(job, logger)
            return job
        if status in ("failed", "cancelled"):
            _log_terminal(job, logger)
            _raise_if_failed(job)
            raise PikaError(f"job {status}")
    raise PikaError(f"job {job_id} did not finish within {int(timeout)}s")


async def cancel_job(client: httpx.AsyncClient, api_key: str, job_id: str, logger: logging.Logger) -> None:
    try:
        resp = await client.post(f"{BASE_URL}/v1/media/jobs/{job_id}/cancel", headers=auth_headers(api_key))
        logger.info(f"Cancel {job_id}: HTTP {resp.status_code}")
    except httpx.HTTPError as e:
        logger.warning(f"Cancel failed: {e}")


async def download_file(client: httpx.AsyncClient, url: str, output_path: str,
                        logger: logging.Logger, attempts: int = 3) -> str:
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            async with client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(1 << 16):
                        f.write(chunk)
            return output_path
        except httpx.HTTPError as e:
            last = e
            logger.warning(f"Download attempt {attempt} failed: {e}")
            await asyncio.sleep(2 * attempt)
    raise PikaError(f"download failed: {last}")


def probe_media(path: str) -> dict:
    """ffprobe a video or audio file. Returns width, height, fps, seconds, sample_rate, channels."""
    info: dict = {}
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return info
        data = json.loads(result.stdout)
        fmt = data.get("format") or {}
        if fmt.get("duration"):
            info["seconds"] = float(fmt["duration"])
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and "width" not in info:
                info["width"] = int(s.get("width") or 0)
                info["height"] = int(s.get("height") or 0)
                num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
                try:
                    info["fps"] = (int(num) / int(den)) if int(den) else 0.0
                except ValueError:
                    info["fps"] = 0.0
                if s.get("duration"):
                    info["seconds"] = float(s["duration"])
            elif s.get("codec_type") == "audio" and "sample_rate" not in info:
                info["sample_rate"] = int(s.get("sample_rate") or 0)
                info["channels"] = int(s.get("channels") or 0)
                if "seconds" not in info and s.get("duration"):
                    info["seconds"] = float(s["duration"])
    except Exception:
        pass
    return info


def charge_micro_usd(job: dict) -> Optional[int]:
    billing = job.get("billing") or {}
    if billing.get("state") == "settled":
        return int(billing.get("charge_micro_usd") or 0)
    return None


def stem_seconds(stem: dict) -> float:
    try:
        n = float(stem.get("num_samples") or 0)
        sr = float(stem.get("sample_rate") or 0)
        return n / sr if sr else 0.0
    except (TypeError, ValueError):
        return 0.0


# --- internals -------------------------------------------------------------

async def _request_with_retry(client, method, url, *, headers, json, logger,
                              attempts: int = 4, retry_only_with_retry_after: bool = False):
    last_resp = None
    for attempt in range(1, attempts + 1):
        try:
            resp = await client.request(method, url, headers=headers, json=json)
        except httpx.HTTPError as e:
            if attempt == attempts:
                raise PikaError(f"network error: {e}")
            logger.warning(f"{method} {url} network error (attempt {attempt}): {e}")
            await asyncio.sleep(2 * attempt)
            continue
        if resp.status_code < 400:
            return resp
        last_resp = resp
        retry_after = resp.headers.get("Retry-After")
        retryable = resp.status_code in RETRYABLE_STATUSES
        if retry_only_with_retry_after:
            retryable = retryable and retry_after is not None
        if not retryable or attempt == attempts:
            break
        wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0 * attempt
        logger.warning(f"{method} {url} HTTP {resp.status_code}, retrying in {wait:.0f}s")
        await asyncio.sleep(min(wait, 60))
    # Non-2xx: either {"message"} or a failed job envelope.
    try:
        body = last_resp.json()
    except Exception:
        body = None
    if isinstance(body, dict) and body.get("status") == "failed":
        _raise_if_failed(body, last_resp.status_code)
    raise PikaError(_extract_message(last_resp), last_resp.status_code)


def _raise_if_failed(job: dict, status_code: int = 0) -> None:
    if job.get("status") == "failed":
        err = job.get("error") or {}
        raise PikaError(err.get("message") or "job failed", status_code, err.get("code") or "")


def _extract_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return data.get("message") or (data.get("error") or {}).get("message") or str(data)[:300]
        return str(data)[:300]
    except Exception:
        return resp.text[:300]


async def _settle_billing(client, api_key, job_id, job, logger) -> dict:
    """billing.state may still be pending right after completion; re-read a few times."""
    for wait in (1, 2, 5):
        billing = job.get("billing") or {}
        if billing.get("state") != "pending":
            return job
        await asyncio.sleep(wait)
        try:
            resp = await client.get(f"{BASE_URL}/v1/media/jobs/{job_id}", headers=auth_headers(api_key))
            if resp.status_code == 200:
                job = resp.json()
        except httpx.HTTPError:
            pass
    return job


def _log_terminal(job: dict, logger: logging.Logger) -> None:
    usage = job.get("usage")
    billing = job.get("billing")
    timing = {k: job.get(k) for k in ("mode", "runtime_s", "inference_s", "queue_s") if job.get(k) is not None}
    logger.info(f"Job terminal: usage={usage} billing={billing} timing={timing}")
