"""Shared HTTP helper for the infsh/grafana app.

One client for the whole Grafana surface: the Grafana API itself (alerts,
silences, annotations, datasources) and anything reachable through its
datasource proxy, Loki included.

Kept as a separate module so the app file stays about Grafana's endpoints
rather than HTTP mechanics. print() is used instead of a module-level logger:
a logger here would not reach task logs.
"""

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

RETRY_STATUSES = (429, 502, 503, 504)
MAX_ATTEMPTS = 4

_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d|w)$")
_DURATION_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def get_api_key() -> str:
    """The Grafana token, from the team's own GRAFANA_API_KEY secret.

    Deliberately not an app input: the platform already stores secrets per
    team, so each team points this app at its own Grafana with its own token
    and no credential travels through a task payload.
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


def resolve_base_url(base_url: Optional[str]) -> str:
    """Which Grafana to talk to: the caller's, else the GRAFANA_URL default."""
    candidate = (base_url or os.environ.get("GRAFANA_URL") or "").strip().rstrip("/")
    if not candidate:
        raise RuntimeError(
            "No Grafana URL. Pass base_url on the request, or set a GRAFANA_URL "
            "default in the app's env."
        )
    if not candidate.startswith(("http://", "https://")):
        raise ValueError(f"base_url must start with http:// or https://, got {candidate!r}")
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
        # (base_url, datasource) -> proxy path prefix. Keyed, so one caller's
        # Grafana can never answer for another's.
        self._datasource_paths: dict[tuple[str, str], str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        base_url: Optional[str] = None,
        params: Optional[Any] = None,
        json_body: Optional[Any] = None,
    ) -> Any:
        """Call the Grafana API, retrying the statuses it documents as transient.

        Returns parsed JSON, or None for an empty body (DELETE returns one).
        """
        url = f"{resolve_base_url(base_url)}{path}"
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

    async def datasource_path(self, datasource: str, base_url: Optional[str]) -> str:
        """Proxy path prefix for a datasource, by uid, name or numeric id.

        Datasource ids differ per Grafana instance, so nothing here may be
        hardcoded: the uid is asked for by name when it is not already one.
        """
        resolved_base = resolve_base_url(base_url)
        wanted = (datasource or "").strip()
        if not wanted:
            raise ValueError("datasource must be a uid, a name, or a numeric id")

        cache_key = (resolved_base, wanted)
        if cache_key in self._datasource_paths:
            return self._datasource_paths[cache_key]

        if wanted.isdigit():
            path = f"/api/datasources/proxy/{wanted}"
        else:
            path = f"/api/datasources/proxy/uid/{await self._uid_for(wanted, resolved_base)}"

        self._datasource_paths[cache_key] = path
        return path

    async def _uid_for(self, wanted: str, base_url: str) -> str:
        """Match a datasource by uid first, then by name, case-insensitively."""
        datasources = await self.request("GET", "/api/datasources", base_url=base_url)
        available = []
        for item in datasources or []:
            uid, name = str(item.get("uid") or ""), str(item.get("name") or "")
            available.append(f"{name} (uid {uid}, type {item.get('type')})")
            if wanted == uid or wanted.lower() == name.lower():
                return uid
        raise RuntimeError(
            f"No datasource {wanted!r} on {base_url}. Available: {'; '.join(available) or 'none'}"
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
