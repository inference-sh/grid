"""HTTP helper for the infsh/pagerduty app.

Events API v2 only: /v2/enqueue for alerts, /v2/change/enqueue for change
events. The routing key comes from the team's secret and is injected here, so
no function has to handle it.

print() is used instead of a module-level logger, which would not reach task
logs from a helper.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

# 429 is documented as retryable; 5xx means the event was not enqueued.
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4

DEFAULT_EVENTS_URL = "https://events.pagerduty.com"


def get_routing_key() -> str:
    """The Events API v2 integration key, from the team's PAGERDUTY_KEY secret.

    Deliberately not an app input: a routing key is a credential, and the
    platform already stores one per team.
    """
    key = os.environ.get("PAGERDUTY_KEY")
    if not key:
        raise RuntimeError(
            "PAGERDUTY_KEY is not set. This app reads the Events API v2 integration key "
            "from your team's secret of that name — find it on the PagerDuty service's "
            "Integrations tab and set it with `belt secrets set PAGERDUTY_KEY <key>`. A "
            "secret whose record exists but holds an empty value is not injected at all."
        )
    return key.strip()


def resolve_base_url() -> str:
    """The Events API base, from the PAGERDUTY_EVENTS_URL secret or app env.

    Never a request field: the routing key goes wherever this points. Set the
    secret to https://events.eu.pagerduty.com for the EU service region.
    """
    candidate = (os.environ.get("PAGERDUTY_EVENTS_URL") or DEFAULT_EVENTS_URL).strip().rstrip("/")
    if not candidate.startswith(("http://", "https://")):
        raise ValueError(f"PAGERDUTY_EVENTS_URL must start with http:// or https://, got {candidate!r}")
    return candidate


def parse_rfc3339(value: str, field: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an RFC3339 timestamp such as 2026-09-24T08:00:00Z, got {value!r}"
        ) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PagerDutyClient:
    def __init__(self, cancelled=None, timeout: float = 60.0):
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cancelled = cancelled or (lambda: False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def enqueue(self, body: Dict[str, Any]) -> Any:
        """Send an alert event (trigger, acknowledge or resolve)."""
        return await self._post("/v2/enqueue", body)

    async def enqueue_change(self, body: Dict[str, Any]) -> Any:
        """Send a change event. These never page."""
        return await self._post("/v2/change/enqueue", body)

    async def _post(self, path: str, body: Dict[str, Any]) -> Any:
        url = f"{resolve_base_url()}{path}"
        payload = {**body, "routing_key": get_routing_key()}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = await self._client.post(url, json=payload)
            if (
                response.status_code in RETRY_STATUSES
                and attempt < MAX_ATTEMPTS
                and not self._cancelled()
            ):
                delay = _retry_after(response, attempt)
                print(
                    f"pagerduty returned {response.status_code}; "
                    f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                continue
            return _result(response, url)

        raise RuntimeError(f"PagerDuty request to {url} exhausted {MAX_ATTEMPTS} attempts")


def _retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return min(2 ** (attempt - 1), 30)


def _result(response: httpx.Response, url: str) -> Any:
    if response.status_code in (400, 401, 403):
        # PagerDuty answers a bad or unknown routing key with 400, not 401.
        raise RuntimeError(
            f"PagerDuty rejected the event ({response.status_code}) at {url}: "
            f"{response.text[:600]}. A 400 here usually means the PAGERDUTY_KEY secret is "
            "not an Events API v2 integration key for the target service."
        )
    if response.status_code >= 400:
        raise RuntimeError(f"PagerDuty API error {response.status_code}: {response.text[:1000]}")
    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text[:2000]}
