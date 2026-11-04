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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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


class ResponseTooLarge(RuntimeError):
    """Loki's answer was bigger than the gRPC message limit between its components.

    Grafana relays it as a 500 whose body reads, verbatim from production:
    `rpc error: code = ResourceExhausted desc = trying to send message larger
    than max (9279245 vs. 4194304)`. `size` is the first number when present.
    """

    def __init__(self, message: str, size: Optional[int] = None):
        super().__init__(message)
        self.size = size


_TOO_LARGE_RE = re.compile(r"larger than max \((\d+) vs\.? ?(\d+)\)")
_TOO_LARGE_MARKERS = ("resourceexhausted", "larger than max", "message too large")


def too_large(body: str) -> Optional[ResponseTooLarge]:
    """A ResponseTooLarge for an error body that reports the size limit, else None."""
    lowered = (body or "").lower()
    if not any(marker in lowered for marker in _TOO_LARGE_MARKERS):
        return None
    match = _TOO_LARGE_RE.search(body)
    return ResponseTooLarge(body, int(match.group(1)) if match else None)


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
        message = f"Grafana API error {response.status_code}: {response.text[:1000]}"
        oversized = too_large(response.text[:1000])
        if oversized is not None:
            raise ResponseTooLarge(message, oversized.size)
        raise RuntimeError(message)
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text[:2000]}


# --- reading a window in pages -----------------------------------------------
#
# Two limits bound one query_range. Loki returns at most LOKI_MAX_ENTRIES lines
# (max_entries_limit_per_query), and its components pass results to each other
# over gRPC with a 4 MiB message cap: a request for 200 lines failed in
# production because those lines averaged 72 KB. So a window is read in pages
# sized by bytes, not only by lines: each page asks for as many lines as fit in
# TARGET_PAGE_BYTES at the line size seen so far, and a page Loki refuses as
# too large is asked again for fewer lines.

LOKI_MAX_ENTRIES = 5000
TARGET_PAGE_BYTES = 3 * 1024 * 1024
# Per-line cost beyond the text: timestamp, framing, and the stream labels when
# every line arrives on its own stream (~250 bytes here: job, filename, ...).
LINE_OVERHEAD_BYTES = 256
# Line size assumed before any page has been seen, when nothing bounds it.
DEFAULT_LINE_BYTES = 1024
# log_digest has Loki cut lines to this many bytes. Grouping reads the first
# 400 characters of a message and examples are at most 4000, so the rest of a
# 72 KB SQL dump only cost pages.
DIGEST_LINE_BYTES = 8192
# Fewest lines a page is shrunk to. A page that is still too large at one line
# holds a single line over 4 MiB, which no page size can read.
MIN_PAGE_LINES = 1


def truncate_in_loki(query: str, keep: int) -> str:
    """The query with Loki cutting every line to `keep` bytes before answering.

    Only for log queries (they start with a stream selector); a metric query
    has no lines to cut and would not parse with the stage appended.
    """
    if not query.lstrip().startswith("{"):
        return query
    return f'{query}\n| line_format "{{{{ __line__ | trunc {keep} }}}}"'


def lines_per_page(bytes_per_line: int) -> int:
    """Lines one request may ask for so its answer stays under TARGET_PAGE_BYTES."""
    fit = TARGET_PAGE_BYTES // max(1, bytes_per_line)
    return max(MIN_PAGE_LINES, min(LOKI_MAX_ENTRIES, fit))


@dataclass
class PageStats:
    pages: int = 0  # requests Loki answered
    shrunk: int = 0  # requests refused as too large and asked again smaller
    page_limit: int = 0  # fewest lines asked for by a request that succeeded
    partial: bool = False  # stopped early: a single line was over the limit
    lines_scanned: int = 0  # Loki's totalLinesProcessed, summed over requests


async def read_range(
    client: "GrafanaClient",
    path: str,
    query: str,
    start: datetime,
    end: datetime,
    limit: int,
    direction: str = "backward",
    bytes_per_line: int = DEFAULT_LINE_BYTES,
) -> Tuple[List[tuple], PageStats]:
    """Up to `limit` (nanos, text, labels) entries of a window, in Loki's order.

    Pages walk from the end of the window backward (or from the start forward),
    each one bounded at the oldest (newest) line of the last. Loki's start is
    inclusive and its end exclusive, so consecutive pages share the boundary
    nanosecond; entries already read are skipped, and one extra slot per page
    makes room for them.
    """
    stats = PageStats()
    start_ns, end_ns = int(to_nanos(start)), int(to_nanos(end))
    seen: set = set()
    entries: List[tuple] = []
    remaining = limit
    overlap = 0
    # After a refusal, pages grow back at most twofold per success: a window
    # of small lines over huge ones would otherwise swing back to full pages
    # and be refused again on every page.
    ceiling: Optional[int] = None
    while remaining > 0:
        page = min(remaining + overlap, lines_per_page(bytes_per_line), ceiling or LOKI_MAX_ENTRIES)
        # The boundary line of the last page comes back first; it needs its own slot.
        floor = MIN_PAGE_LINES + overlap
        page = max(page, floor)
        try:
            payload = await client.request(
                "GET",
                path,
                params={
                    "query": query,
                    "start": str(start_ns),
                    "end": str(end_ns),
                    "limit": page,
                    "direction": direction,
                },
            )
        except ResponseTooLarge as exc:
            if page <= floor:
                if entries:
                    print(f"loki page of {page} lines still too large ({exc.size} bytes); stopping")
                    stats.partial = True
                    break
                raise RuntimeError(
                    f"Loki's answer for {page} line(s) is larger than Grafana's 4 MiB gRPC "
                    f"message limit ({exc.size or 'unknown'} bytes): the newest line in the "
                    "window is too large to return. Narrow the window or add filters to "
                    f"the query. Grafana said: {exc}"
                ) from exc
            smaller = page // 2
            if exc.size:
                # The reported size says how far over the page was. It can only
                # overstate how many lines fit (the window may hold fewer lines
                # than were asked for), so halving still applies as a ceiling.
                smaller = min(smaller, page * TARGET_PAGE_BYTES // exc.size)
            smaller = max(floor, smaller)
            bytes_per_line = -(-TARGET_PAGE_BYTES // smaller)
            ceiling = smaller
            stats.shrunk += 1
            print(f"loki page of {page} lines too large ({exc.size} bytes); retrying with {smaller}")
            continue

        if ceiling is not None:
            ceiling = page * 2
        stats.pages += 1
        stats.page_limit = page if not stats.page_limit else min(stats.page_limit, page)
        data = (payload or {}).get("data") or {}
        stats.lines_scanned += lines_scanned(data)
        returned, added, size, edge = 0, 0, 0, None
        for stream in data.get("result") or []:
            labels = stream.get("stream") or {}
            label_key = tuple(sorted(labels.items()))
            for item in stream.get("values") or []:
                returned += 1
                nanos, text = int(item[0]), item[1]
                size += len(text.encode("utf-8", "replace")) + LINE_OVERHEAD_BYTES
                if edge is None:
                    edge = nanos
                elif direction == "backward":
                    edge = min(edge, nanos)
                else:
                    edge = max(edge, nanos)
                key = (nanos, text, label_key)
                if key in seen:
                    continue
                seen.add(key)
                entries.append((nanos, text, labels))
                added += 1
        if returned:
            bytes_per_line = max(1, size // returned)
        remaining -= added
        if returned < page or added == 0 or edge is None:
            break
        if direction == "backward":
            end_ns = edge + 1
        else:
            start_ns = edge
        overlap = 1
    return entries, stats


def lines_scanned(data: Dict[str, Any]) -> int:
    summary = ((data.get("stats") or {}).get("summary")) or {}
    try:
        return int(summary.get("totalLinesProcessed") or 0)
    except (TypeError, ValueError):
        return 0


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


_JSON_PAIR_RE = re.compile(
    r'"([A-Za-z_][\w.\-]*)"\s*:\s*'
    r'("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
)
_JSON_TAIL_RE = re.compile(r'"([A-Za-z_][\w.\-]*)"\s*:\s*"((?:[^"\\]|\\.)*)\\?$')


def parse_record(text: str, cut: bool = False) -> Dict[str, Any]:
    """Best-effort structured view of a log line. Plain text stays plain text.

    A JSON line that was `cut` short no longer parses; its scalar fields are
    read up to the cut instead, the last one possibly partial. Producers put
    level, component, time and message first, so grouping still sees them.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            if cut:
                salvaged = _salvage_json(stripped)
                if salvaged:
                    return salvaged
    return {"message": text}


def _salvage_json(text: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    end = 0
    for match in _JSON_PAIR_RE.finditer(text):
        try:
            record.setdefault(match.group(1), json.loads(match.group(2)))
        except ValueError:
            continue
        end = match.end()
    tail = _JSON_TAIL_RE.search(text, end)
    if tail:
        try:
            record.setdefault(tail.group(1), json.loads(f'"{tail.group(2)}"'))
        except ValueError:
            pass
    return record


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
