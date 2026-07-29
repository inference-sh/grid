"""
Shared helper for the Mirage (formerly Captions) APIs.
Symlink this file into each app folder for deployment.

Two hosts are live under one brand and one API key:

  MIRAGE_BASE   https://api.mirage.app/v1   — the current platform. Video generation
                (image + audio -> talking head) and video captions. multipart
                requests, GET polling by video id, content served via redirect.

  LEGACY_BASE   https://api.captions.ai/api — the original Captions API. AI Creator,
                AI Ads and AI Twin. JSON requests, submit -> poll by operationId,
                result handed back as a plain URL.

Both authenticate with the CAPTIONS_KEY secret in an `x-api-key` header.
"""

import os
import asyncio
import logging
import re
import tempfile
import uuid
import httpx

logger = logging.getLogger(__name__)

MIRAGE_BASE = "https://api.mirage.app/v1"
LEGACY_BASE = "https://api.captions.ai/api"

POLL_INTERVAL = 3.0
MAX_POLL_TIME = 1800


def get_api_key() -> str:
    api_key = os.environ.get("CAPTIONS_KEY")
    if not api_key:
        raise RuntimeError(
            "CAPTIONS_KEY secret is not set. A secret whose record exists but holds an "
            "empty value is not injected into the container at all — check that "
            "`belt secrets get CAPTIONS_KEY --json` reports a non-empty masked_value, "
            "and re-set it with `belt secrets set CAPTIONS_KEY <key>` if it does not. "
            "Keys come from platform.mirage.app."
        )
    return api_key.strip()


def log_key_fingerprint(log=None) -> None:
    """Log a non-revealing shape check on the key. Never raises.

    Pass the app's `self.logger` — the module-level logger here is not wired
    into the kernel's log capture, so its output never reaches task logs.

    Called from setup() so a misconfigured secret is visible in the logs
    without taking the worker down before it can report anything. A value
    that is absent, empty, whitespace-padded or URL-shaped points at a
    mis-set secret rather than at an API problem — worth distinguishing
    before reading a 401 as an entitlement issue.
    """
    log = log or logger
    raw = os.environ.get("CAPTIONS_KEY")
    if raw is None:
        present = sorted(k for k in os.environ if "CAPTION" in k.upper() or k.endswith("_KEY"))
        log.error(
            "CAPTIONS_KEY is absent from the environment — the secret was not injected. "
            f"Key-shaped env vars actually present: {present}"
        )
        return
    log.info(
        f"CAPTIONS_KEY present: len={len(raw)} empty={raw == ''} "
        f"looks_like_url={raw.startswith('http')} padded={raw.strip() != raw}"
    )


def get_client(timeout: float = 180) -> httpx.AsyncClient:
    """Client for the current Mirage platform (api.mirage.app).

    No Content-Type is set: both write endpoints are multipart, so httpx
    must be left to build the boundary itself.
    """
    return httpx.AsyncClient(
        base_url=MIRAGE_BASE,
        timeout=timeout,
        headers={"x-api-key": get_api_key()},
        follow_redirects=True,
    )


def get_legacy_client(timeout: float = 180) -> httpx.AsyncClient:
    """Client for the legacy Captions API (api.captions.ai)."""
    return httpx.AsyncClient(
        base_url=LEGACY_BASE,
        timeout=timeout,
        headers={
            "x-api-key": get_api_key(),
            "Content-Type": "application/json",
        },
        follow_redirects=True,
    )


def _raise_for_error(resp: httpx.Response, what: str) -> None:
    if resp.status_code < 400:
        return
    body = resp.text[:800]
    logger.error(f"{what} -> HTTP {resp.status_code}: {body}")
    detail = body
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                detail = err.get("message", detail)
            elif isinstance(payload.get("detail"), str):
                detail = payload["detail"]
    except ValueError:
        pass
    if resp.status_code == 401:
        raise RuntimeError(
            f"Mirage API rejected the API key (401) on {what}. "
            f"Keys are issued per platform — a key from platform.mirage.app may not "
            f"work against api.captions.ai and vice versa. Detail: {detail}"
        )
    if resp.status_code == 429:
        raise RuntimeError(
            f"Mirage API rate limit hit on {what}. Video generation is limited to "
            f"2 requests/min per organization. Detail: {detail}"
        )
    raise RuntimeError(f"Mirage API error on {what} ({resp.status_code}): {detail}")


# --------------------------------------------------------------------------
# api.mirage.app  — multipart submit, GET poll, redirect download
# --------------------------------------------------------------------------


async def post_multipart(
    client: httpx.AsyncClient, path: str, data: dict, files: dict | None = None
) -> dict:
    """POST multipart/form-data to the Mirage platform and return the JSON body.

    data must be a dict. To send an ordered repeated field (texts, fonts, sizes,
    colors on the text-overlay endpoint) give the key a list value — httpx emits
    one form part per entry, preserving order. Do not pass a list of (key, value)
    pairs: httpx treats any non-dict data as raw request content and fails with
    "Attempted to send an sync request with an AsyncClient instance".
    """
    logger.info(f"POST {MIRAGE_BASE}{path} fields={sorted(data)} files={sorted(files or {})}")
    resp = await client.post(path, data=data, files=files or None)
    _raise_for_error(resp, f"POST {path}")
    return resp.json()


async def get_json(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict:
    resp = await client.get(path, params=params)
    _raise_for_error(resp, f"GET {path}")
    return resp.json()


async def poll_video(client: httpx.AsyncClient, video_id: str) -> dict:
    """Poll GET /videos/{id} until COMPLETE, or raise on FAILED/CANCELLED."""
    elapsed = 0.0
    last_progress = None
    while elapsed < MAX_POLL_TIME:
        data = await get_json(client, f"/videos/{video_id}")
        status = data.get("status", "")
        progress = data.get("progress")
        if (status, progress) != (None, last_progress):
            logger.info(f"video {video_id}: {status} progress={progress} ({elapsed:.0f}s)")
        last_progress = progress

        if status == "COMPLETE":
            return data
        if status in ("FAILED", "CANCELLED"):
            err = data.get("error") or {}
            raise RuntimeError(
                f"Mirage job {video_id} ended {status}: "
                f"[{err.get('code', 'unknown')}] {err.get('message', 'no detail')}"
            )

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Mirage job {video_id} did not finish within {MAX_POLL_TIME}s")


async def download_video_content(client: httpx.AsyncClient, video_id: str) -> str:
    """GET /videos/{id}/content (302 -> CDN) and save the mp4 to a temp path."""
    logger.info(f"downloading content for {video_id}")
    resp = await client.get(f"/videos/{video_id}/content")
    _raise_for_error(resp, f"GET /videos/{video_id}/content")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(resp.content)
        path = f.name
    logger.info(f"saved {len(resp.content)} bytes to {path}")
    return path


async def poll_text_overlay(client: httpx.AsyncClient, overlay_id: str) -> dict:
    """Poll GET /meta/text_overlays/{id} until the job reaches a terminal state.

    Unlike a video job this fans out: the job carries a results[] with one entry
    per input text, each with its own COMPLETE/FAILED status. A job can finish
    COMPLETE with individual items failed, so the caller must inspect results[]
    rather than trusting the job status alone.
    """
    elapsed = 0.0
    while elapsed < MAX_POLL_TIME:
        data = await get_json(client, f"/meta/text_overlays/{overlay_id}")
        status = data.get("status", "")
        results = data.get("results") or []
        logger.info(
            f"overlay {overlay_id}: {status} "
            f"({sum(1 for r in results if r.get('status') == 'COMPLETE')}/{len(results)} done, "
            f"{elapsed:.0f}s)"
        )

        if status == "COMPLETE":
            return data
        if status in ("FAILED", "CANCELLED"):
            err = data.get("error") or {}
            raise RuntimeError(
                f"Mirage overlay job {overlay_id} ended {status}: "
                f"[{err.get('code', 'unknown')}] {err.get('message', 'no detail')}"
            )

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Mirage overlay job {overlay_id} did not finish within {MAX_POLL_TIME}s")


async def list_caption_templates(
    client: httpx.AsyncClient, limit: int = 100, after: str | None = None
) -> list[dict]:
    """List caption style templates, following pagination up to `limit` items."""
    out: list[dict] = []
    cursor = after
    while len(out) < limit:
        page = await get_json(
            client,
            "/videos/captions/templates",
            params={"limit": min(100, limit - len(out)), **({"after": cursor} if cursor else {})},
        )
        items = page.get("data", [])
        out.extend(items)
        if not page.get("has_more") or not items:
            break
        cursor = items[-1]["id"]
    return out[:limit]


# --------------------------------------------------------------------------
# api.captions.ai  — JSON submit, operationId poll, result URL
# --------------------------------------------------------------------------


async def post_legacy(client: httpx.AsyncClient, path: str, payload: dict) -> dict:
    """POST JSON to the legacy Captions API and return the JSON body."""
    logger.info(f"POST {LEGACY_BASE}{path} keys={sorted(payload)}")
    resp = await client.post(path, json=payload)
    _raise_for_error(resp, f"POST {path}")
    return resp.json()


async def poll_legacy(client: httpx.AsyncClient, path: str, operation_id: str) -> dict:
    """Poll a legacy poll/status endpoint until it returns a url or COMPLETE.

    The legacy poll response is a union: either {"state": ..., "progress": ...}
    while running, or a body carrying "url" once finished. `/twin/status`
    reports COMPLETE with no url, since the product of that job is the twin
    itself rather than a file.
    """
    elapsed = 0.0
    while elapsed < MAX_POLL_TIME:
        data = await post_legacy(client, path, {"operationId": operation_id})
        state = data.get("state", "")
        logger.info(
            f"operation {operation_id}: {state or 'no state'} "
            f"progress={data.get('progress')} ({elapsed:.0f}s)"
        )

        if data.get("url"):
            return data
        if state == "COMPLETE":
            return data
        if state in ("FAILED", "ERROR", "CANCELLED"):
            raise RuntimeError(
                f"Captions operation {operation_id} ended {state}: "
                f"{data.get('message') or data.get('detail') or 'no detail'}"
            )

        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Captions operation {operation_id} did not finish within {MAX_POLL_TIME}s")


def meta_get(metadata, key: str) -> str:
    """Read one field from run()'s metadata argument, whatever shape it has.

    Deployed kernels differ: older ones hand run() the raw request dict (and
    currently an empty one), newer ones hand it a Metadata object that also
    exposes .get() for compatibility. Treating it as an object only, or as a
    dict only, silently yields nothing on the other half of the fleet.
    """
    if metadata is None:
        return ""
    if isinstance(metadata, dict):
        return (metadata.get(key) or "").strip()
    getter = getattr(metadata, "get", None)
    if callable(getter):
        try:
            return (getter(key) or "").strip()
        except Exception:
            pass
    return (getattr(metadata, key, None) or "").strip()


_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def scoped_name(metadata, label: str, log=None) -> str:
    """Build an unguessable name for a twin created under the shared API key.

    What this protects against: creator_name on /creator/submit and /ads/submit
    is a free-form string, so a twin named "spokesperson" could be performed by
    anyone who guesses that word. A random component makes the name impossible
    to reach without having been handed it.

    The random component is what provides that — team_id and task_id are folded
    in only for traceability, so a twin in the provider's account can be tied
    back to a tenant and a run. Both are set by the engine rather than the
    caller, and both are frequently absent (deployed workers currently receive
    an empty metadata dict), so neither is relied on.

    Workers are reused across tenants: call this per request from run() and
    never store the result on the app instance.
    """
    team_id = meta_get(metadata, "team_id")
    task_id = meta_get(metadata, "task_id")
    if log and not (team_id and task_id):
        log.info(
            f"engine metadata partial (team_id={team_id or 'absent'}, "
            f"task_id={task_id or 'absent'}); name stays unguessable via its "
            f"random component, but is less traceable"
        )
    safe = _NAME_SAFE.sub("-", label).strip("-")[:32] or "twin"
    parts = [p for p in (team_id, task_id, uuid.uuid4().hex[:12], safe) if p]
    return "-".join(parts)


async def download_url(url: str, suffix: str = ".mp4") -> str:
    """Download a result URL to a temp path."""
    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as dl:
        resp = await dl.get(url)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(resp.content)
            path = f.name
    logger.info(f"downloaded {len(resp.content)} bytes to {path}")
    return path


def probe_audio_seconds(path: str) -> float:
    """Duration of an audio file in seconds via ffprobe, 0.0 if it can't be read.

    The SDK ships probe_video but no audio equivalent, and audio duration is
    what drives the generated video's length (and therefore its price).
    """
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return float(json.loads(result.stdout).get("format", {}).get("duration", 0.0))
    except Exception:
        logger.warning(f"could not probe audio duration for {path}")
    return 0.0


def billed_seconds(seconds: float, increment: int = 6) -> int:
    """Round up to the provider's billing increment.

    Mirage Video bills the generated output in 6-second increments at
    $0.175/sec, so a 10.2s result is charged as 12s.
    """
    if seconds <= 0:
        return 0
    return int(-(-seconds // increment) * increment)


def public_url(file_obj, what: str) -> str:
    """Extract a publicly reachable URL from an inferencesh File.

    The legacy AI Ads and AI Twin endpoints take URLs rather than uploads,
    so a local-only file cannot be passed through.
    """
    if file_obj.uri and file_obj.uri.startswith("http"):
        return file_obj.uri
    raise RuntimeError(
        f"{what} must be a publicly accessible URL — the Captions AI Ads/Twin "
        f"endpoints fetch media by URL and cannot accept a local upload."
    )
