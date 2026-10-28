"""Tencent Cloud International's `hunyuan` service, shared by the tencent apps: signed calls, job polling, result files.

Every action is a TC3-HMAC-SHA256 signed `POST /` with a SecretId and
SecretKey. Jobs are submitted, then polled until `Status` is DONE or FAIL;
results are `ResultFile3Ds` of `{Type, Url, PreviewImageUrl}` valid 24 hours.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import httpx
from inferencesh import File

HOST = "hunyuan.intl.tencentcloudapi.com"
SERVICE = "hunyuan"
VERSION = "2023-09-01"
SECRET_ID = "TENCENTCLOUD_SECRET_ID"
SECRET_KEY = "TENCENTCLOUD_SECRET_KEY"

MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
# Engine-side and rate-limit errors Tencent says to retry later.
RETRY_CODES = {"RequestLimitExceeded", "FailedOperation.EngineServerLimitExceeded", "InternalError"}
JOB_TIMEOUT = 30 * 60
# A submit while the account's jobs are all running (one by default) is refused with this code: wait for a slot.
JOB_SLOT_CODE = "RequestLimitExceeded.JobNumExceed"
JOB_SLOT_WAIT = 20 * 60
JOB_SLOT_POLL = 15
POLL_INTERVAL = 5
NOT_MODELS = ("GIF", "IMAGE", "MP4")

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Tencent Cloud answered with an error, or a job failed. The message carries Tencent's code."""


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def sign(secret_id: str, secret_key: str, action: str, region: str, payload: str, timestamp: int) -> dict:
    """Headers for one TC3-HMAC-SHA256 signed POST (https://www.tencentcloud.com/document/api/1284/75533)."""
    date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")
    content_type = "application/json; charset=utf-8"
    canonical = "POST\n/\n\ncontent-type:%s\nhost:%s\n\ncontent-type;host\n%s" % (
        content_type, HOST, hashlib.sha256(payload.encode("utf-8")).hexdigest()
    )
    scope = "%s/%s/tc3_request" % (date, SERVICE)
    to_sign = "TC3-HMAC-SHA256\n%d\n%s\n%s" % (timestamp, scope, hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    key = _hmac(_hmac(_hmac(("TC3" + secret_key).encode("utf-8"), date), SERVICE), "tc3_request")
    signature = hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "Authorization": "TC3-HMAC-SHA256 Credential=%s/%s, SignedHeaders=content-type;host, Signature=%s" % (secret_id, scope, signature),
        "Content-Type": content_type,
        "Host": HOST,
        "X-TC-Action": action,
        "X-TC-Version": VERSION,
        "X-TC-Region": region,
        "X-TC-Timestamp": str(timestamp),
    }


class TencentClient:
    def __init__(self):
        self.secret_id = os.environ.get(SECRET_ID)
        self.secret_key = os.environ.get(SECRET_KEY)
        if not self.secret_id or not self.secret_key:
            raise ValueError(f"{SECRET_ID} and {SECRET_KEY} are not set: connect the Tencent Cloud credential")
        self.region = os.environ.get("TENCENTCLOUD_REGION", "ap-singapore")
        self.http = httpx.AsyncClient(base_url="https://" + HOST, timeout=httpx.Timeout(120, connect=15))

    async def call(self, action: str, params: dict) -> dict:
        """One signed action returning `Response`: rate-limit and engine-busy errors retried.

        A submit has no idempotency key; these errors reject the request outright, so resending it is safe.
        """
        payload = json.dumps(params, separators=(",", ":"))
        slot_deadline = time.monotonic() + JOB_SLOT_WAIT
        attempt = 0
        while True:
            headers = sign(self.secret_id, self.secret_key, action, self.region, payload, int(time.time()))
            response = await self.http.post("/", content=payload, headers=headers)
            code, error = None, {}
            if not response.is_error:
                body = response.json().get("Response") or {}
                error = body.get("Error")
                if not error:
                    return body
                code = error.get("Code", "")
            if code == JOB_SLOT_CODE and time.monotonic() < slot_deadline:
                logger.info("%s: all job slots busy, waiting %ds", action, JOB_SLOT_POLL)
                await asyncio.sleep(JOB_SLOT_POLL)
                continue
            if attempt < MAX_RETRIES and (response.status_code in RETRY_STATUSES or code in RETRY_CODES or (code or "").split(".")[0] in RETRY_CODES):
                delay = 2 ** (attempt + 2)
                logger.warning("%s: %s, retrying in %ds", action, code or response.status_code, delay)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            if response.is_error:
                raise UpstreamError("Tencent Cloud %d: %s" % (response.status_code, response.text[:500]))
            raise UpstreamError("Tencent Cloud %s: %s" % (code, error.get("Message", "")))

    async def run_job(self, submit: str, query: str, params: dict, label: str) -> dict:
        """Submit a job and poll it until DONE; raises on FAIL or after JOB_TIMEOUT."""
        job_id = (await self.call(submit, params))["JobId"]
        logger.info("submitted %s job %s (region %s)", label, job_id, self.region)
        started = time.monotonic()
        last = None
        while True:
            job = await self.call(query, {"JobId": job_id})
            status = job.get("Status")
            if status != last:
                logger.info("job %s: %s", job_id, status)
                last = status
            if status == "DONE":
                return job
            if status == "FAIL":
                raise UpstreamError("%s job %s failed: %s %s" % (label, job_id, job.get("ErrorCode", ""), job.get("ErrorMessage", "")))
            if time.monotonic() - started > JOB_TIMEOUT:
                raise UpstreamError("%s job %s did not finish in %d minutes" % (label, job_id, JOB_TIMEOUT // 60))
            await asyncio.sleep(POLL_INTERVAL)

    async def close(self):
        await self.http.aclose()


async def download(url: str, fallback_suffix: str) -> str:
    name = os.path.basename(url.split("?")[0]) or "file" + fallback_suffix
    fd, out_path = tempfile.mkstemp(suffix="_" + name)
    os.close(fd)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
    return out_path


async def result_files(job: dict, label: str) -> Tuple[File, Optional[File], List[File]]:
    """The job's files as (model, preview, others). The model is the GLB when there is one, else the first model file."""
    results = [r for r in job.get("ResultFile3Ds") or [] if r.get("Url")]
    logger.info("files: %s", ", ".join("%s %s" % (r.get("Type"), os.path.basename(r["Url"].split("?")[0])) for r in results))
    models = [r for r in results if (r.get("Type") or "").upper() not in NOT_MODELS]
    primary = next((r for r in models if (r.get("Type") or "").upper() == "GLB"), models[0] if models else None)
    if primary is None:
        raise UpstreamError("%s returned no model: %s" % (label, [r.get("Type") for r in results]))
    model, preview, others = None, None, []
    for result in results:
        path = await download(result["Url"], ".zip")
        if result is primary:
            model = File(path=path)
        else:
            others.append(File(path=path))
        if preview is None and result.get("PreviewImageUrl"):
            preview = File(path=await download(result["PreviewImageUrl"], ".png"))
    return model, preview, others
