"""Shared HTTP helper for the infsh/grafana app.

One client for the whole Grafana surface: the Grafana API itself (alerts,
silences, annotations, datasources) and anything reachable through its
datasource proxy, Loki included.

Kept as a separate module so the app file stays about Grafana's endpoints
rather than HTTP mechanics. print() is used instead of a module-level logger:
a logger here would not reach task logs.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

RETRY_STATUSES = (429, 502, 503, 504)
MAX_ATTEMPTS = 4

_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d|w)$")
_DURATION_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def get_api_key() -> str:
    """The Grafana token, from the team's own GRAFANA_API_KEY secret.

    Deliberately not an app input: the platform already stores secrets per
    team, so each team points this app at its own Grafana (GRAFANA_URL) with
    its own token and no credential travels through a task payload.
    """
    key = os.environ.get("GRAFANA_API_KEY")
    if not key:
        raise RuntimeError(
            "GRAFANA_API_KEY is not set. This app reads the Grafana token from your "
            "team's secret of that name — set it with `belt secrets set GRAFANA_API_KEY "
            "<token>`. A secret whose record exists but holds an empty value is not "
            "injected at all, so check `belt secrets get GRAFANA_API_KEY --json` reports "
            "a non-empty masked_value."
        )
    return key.strip()


def resolve_base_url() -> str:
    """Which Grafana to talk to: the team's GRAFANA_URL secret, else the app env default.

    Never a request field: the token goes wherever this points.
    """
    candidate = (os.environ.get("GRAFANA_URL") or "").strip().rstrip("/")
    if not candidate:
        raise RuntimeError(
            "No Grafana URL. Connect the grafana credential: `belt credentials connect grafana`, "
            "or `belt secrets attach GRAFANA_URL grafana` if the key is already in the vault."
        )
    if not candidate.startswith(("http://", "https://")):
        raise ValueError(f"GRAFANA_URL must start with http:// or https://, got {candidate!r}")
    return candidate


def parse_duration(value: str) -> timedelta:
    """Parse a relative window such as 30s, 15m, 6h, 2d, 1w."""
    match = _DURATION_RE.match((value or "").strip())
    if not match:
        raise ValueError(
            f"invalid duration {value!r}; use a number followed by s, m, h, d or w (e.g. 15m)"
        )
    amount, unit = int(match.group(1)), match.group(2)
    if amount <= 0:
        raise ValueError(f"duration must be positive, got {value!r}")
    return timedelta(seconds=amount * _DURATION_SECONDS[unit])


def resolve_window(
    since: str, start: Optional[str], end: Optional[str]
) -> tuple[datetime, datetime]:
    """Resolve a query window to absolute UTC timestamps.

    Explicit RFC3339 start/end win; otherwise the window is `since` long and
    ends now.
    """
    end_dt = parse_rfc3339(end, "end") if end else datetime.now(timezone.utc)
    start_dt = parse_rfc3339(start, "start") if start else end_dt - parse_duration(since)
    if start_dt >= end_dt:
        raise ValueError("start must be earlier than end")
    return start_dt, end_dt


def parse_rfc3339(value: str, field: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an RFC3339 timestamp such as 2026-09-24T08:00:00Z, got {value!r}"
        ) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def to_nanos(moment: datetime) -> str:
    """Loki wants nanosecond epoch timestamps, as strings."""
    return str(int(moment.timestamp() * 1_000_000_000))


def from_nanos(nanos: Any) -> str:
    try:
        seconds = int(nanos) / 1_000_000_000
    except (TypeError, ValueError):
        return str(nanos)
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def millis(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


class GrafanaClient:
    """Thin async wrapper over the Grafana HTTP API."""

    def __init__(self, cancelled=None, timeout: float = 120.0):
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cancelled = cancelled or (lambda: False)
        # datasource -> proxy path prefix
        self._datasource_paths: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Any] = None,
        json_body: Optional[Any] = None,
    ) -> Any:
        """Call the Grafana API, retrying the statuses it documents as transient.

        Returns parsed JSON, or None for an empty body (DELETE returns one).
        """
        url = f"{resolve_base_url()}{path}"
        headers = {"Authorization": f"Bearer {get_api_key()}"}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = await self._client.request(
                method, url, headers=headers, params=params, json=json_body
            )
            if (
                response.status_code in RETRY_STATUSES
                and attempt < MAX_ATTEMPTS
                and not self._cancelled()
            ):
                delay = _retry_after(response, attempt)
                print(
                    f"grafana returned {response.status_code}; "
                    f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                continue
            return _result(response, method, url)

        raise RuntimeError(f"Grafana request to {url} exhausted {MAX_ATTEMPTS} attempts")

    async def datasource_path(self, datasource: str) -> str:
        """Proxy path prefix for a datasource, by uid, name or numeric id.

        Datasource ids differ per Grafana instance, so nothing here may be
        hardcoded: the uid is asked for by name when it is not already one.
        """
        wanted = (datasource or "").strip()
        if not wanted:
            raise ValueError("datasource must be a uid, a name, or a numeric id")

        if wanted in self._datasource_paths:
            return self._datasource_paths[wanted]

        if wanted.isdigit():
            path = f"/api/datasources/proxy/{wanted}"
        else:
            path = f"/api/datasources/proxy/uid/{await self._uid_for(wanted)}"

        self._datasource_paths[wanted] = path
        return path

    async def _uid_for(self, wanted: str) -> str:
        """Match a datasource by uid first, then by name, case-insensitively."""
        datasources = await self.request("GET", "/api/datasources")
        available = []
        for item in datasources or []:
            uid, name = str(item.get("uid") or ""), str(item.get("name") or "")
            available.append(f"{name} (uid {uid}, type {item.get('type')})")
            if wanted == uid or wanted.lower() == name.lower():
                return uid
        raise RuntimeError(
            f"No datasource {wanted!r} on {resolve_base_url()}. Available: {'; '.join(available) or 'none'}"
        )


def _retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return min(2 ** (attempt - 1), 30)


def _result(response: httpx.Response, method: str, url: str) -> Any:
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Grafana rejected the credentials ({response.status_code}) for {method} {url}. "
            "Check the GRAFANA_API_KEY secret and the service account's permissions."
        )
    if response.status_code == 404:
        raise RuntimeError(f"Grafana has no {method} {url} (404): {response.text[:500]}")
    if response.status_code >= 400:
        raise RuntimeError(f"Grafana API error {response.status_code}: {response.text[:1000]}")
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text[:2000]}


# --- log digest ---------------------------------------------------------------
#
# Counting lines answers "how loud"; grouping answers "how many problems".
# A window where one benign retry loop logs 400 times and a data-loss bug logs
# 20 reads, line by line, as one problem. Collapsed to signatures it reads as
# two, and the reader can see both.

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
# Opaque ids: 16+ chars mixing letters and digits. Catches base36/base62 ids and
# long hex alike, without eating ordinary words.
_ID_RE = re.compile(r"\b(?=[0-9a-z]*\d)(?=[0-9a-z]*[a-z])[0-9a-z]{16,}\b", re.I)
_HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_IPV6_RE = re.compile(r"\b(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}\b", re.I)
_DUR_RE = re.compile(r"\b\d+(?:\.\d+)?(?:ns|µs|us|ms|s|m|h)\b")
_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?(?:B|KB|MB|GB|KiB|MiB|GiB)\b")
# Only 6+ digit numbers. Shorter ones carry meaning — HTTP status, SQLSTATE,
# task outcome codes — and collapsing 403 into 500 merges two different problems.
_NUM_RE = re.compile(r"\b\d{6,}\b")

_MESSAGE_KEYS = ("message", "msg", "error", "err", "event", "reason", "detail")
GROUP_KEYS = ("component", "logger", "caller", "source", "service", "level", "status")
SKIP_DISTINCT = {"level", "time", "timestamp", "ts", "message", "msg"}


def normalize_line(text: str) -> str:
    """Replace the parts of a log line that vary per occurrence."""
    out = _UUID_RE.sub("<id>", text)
    out = _TS_RE.sub("<ts>", out)
    out = _IPV6_RE.sub("<ip>", out)
    out = _IPV4_RE.sub("<ip>", out)
    out = _ID_RE.sub("<id>", out)
    out = _HEX_RE.sub("<id>", out)
    out = _DUR_RE.sub("<dur>", out)
    out = _SIZE_RE.sub("<size>", out)
    out = _NUM_RE.sub("<n>", out)
    return re.sub(r"\s+", " ", out).strip()


def parse_record(text: str) -> Dict[str, Any]:
    """Best-effort structured view of a log line. Plain text stays plain text."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {"message": text}


def record_message(record: Dict[str, Any]) -> str:
    """The human-readable part of a record, whatever the producer called it."""
    for key in _MESSAGE_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested = record_message(value)
            if nested:
                return nested
    method, path = record.get("method"), record.get("path")
    if method and path:
        return f"{method} {path}"
    if path:
        return str(path)
    return ""
