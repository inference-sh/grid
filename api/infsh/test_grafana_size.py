"""infsh/grafana: Loki answers over Grafana's 4 MiB gRPC limit are read in smaller pages.

The app's __init__ uses a relative import, so the app directory is imported as
the `grafana` package from here, one level up.
"""

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grafana import grafana_http as gh  # noqa: E402
from grafana.inference import App, LogDigestInput, QueryLogsInput  # noqa: E402

# Verbatim from task 79x2pg2drezary4ghw0vmgngxg (query_logs, limit 200).
GRAFANA_BODY = (
    "rpc error: code = ResourceExhausted desc = trying to send message larger than max "
    "(9279245 vs. 4194304)"
)
PATH = "/api/datasources/proxy/uid/loki/loki/api/v1/query_range"
BASE = 1_800_000_000_000_000_000


class FakeLoki:
    """Stands in for GrafanaClient.request over a window of `lines` (newest last).

    A request whose answer would be bigger than `max_bytes` raises what the
    real client raises for Grafana's 500, so the size check is the real one.
    """

    def __init__(self, lines, max_bytes=gh.TARGET_PAGE_BYTES + 1024 * 1024, labels=None):
        self.lines = lines  # [(nanos, text)]
        self.max_bytes = max_bytes
        self.labels = labels or {"job": "api"}
        self.calls = []

    async def request(self, method, path, *, params=None, json_body=None):
        self.calls.append(dict(params))
        start, end = int(params["start"]), int(params["end"])
        window = [item for item in self.lines if start <= item[0] < end]
        window.sort(key=lambda item: item[0], reverse=params["direction"] == "backward")
        page = window[: int(params["limit"])]
        size = sum(len(text) for _, text in page)
        if size > self.max_bytes:
            body = (
                "rpc error: code = ResourceExhausted desc = trying to send message larger "
                f"than max ({size} vs. 4194304)"
            )
            raise gh.ResponseTooLarge(f"Grafana API error 500: {body}", size)
        return {
            "data": {
                "result": [
                    {"stream": self.labels, "values": [[str(n), t] for n, t in page]}
                ],
                "stats": {"summary": {"totalLinesProcessed": len(window)}},
            }
        }


def window_of(count, line_bytes):
    return [(BASE + i * 1_000_000, f"line {i} " + "x" * line_bytes) for i in range(count)]


def run(coro):
    return asyncio.run(coro)


def start_end(lines):
    return gh.from_nanos(lines[0][0] - 1_000_000), gh.from_nanos(lines[-1][0] + 1_000_000)


# --- the error Grafana returns ------------------------------------------------


def test_result_raises_response_too_large_with_size():
    response = httpx.Response(500, text=GRAFANA_BODY, request=httpx.Request("GET", "http://g/x"))
    with pytest.raises(gh.ResponseTooLarge) as caught:
        gh._result(response, "GET", "http://g/x")
    assert caught.value.size == 9279245
    assert "Grafana API error 500" in str(caught.value)
    # Still a RuntimeError, so callers that caught that keep working.
    assert isinstance(caught.value, RuntimeError)


def test_other_500_is_plain_runtime_error():
    response = httpx.Response(500, text="too many outstanding requests", request=httpx.Request("GET", "http://g/x"))
    with pytest.raises(RuntimeError) as caught:
        gh._result(response, "GET", "http://g/x")
    assert not isinstance(caught.value, gh.ResponseTooLarge)


def test_too_large_without_numbers():
    found = gh.too_large("rpc error: code = ResourceExhausted desc = grpc: received message larger than max")
    assert found is not None and found.size is None
    assert gh.too_large("context deadline exceeded") is None


# --- read_range ---------------------------------------------------------------


def test_lines_per_page_stays_under_target():
    assert gh.lines_per_page(100) == gh.LOKI_MAX_ENTRIES
    assert gh.lines_per_page(72_000) * 72_000 <= gh.TARGET_PAGE_BYTES
    assert gh.lines_per_page(10_000_000) == gh.MIN_PAGE_LINES == 1


def test_error_then_smaller_page_succeeds():
    # The production shape: 128 lines of 72 KB in the window, 9.2 MB in one answer.
    lines = window_of(128, 72_000)
    loki = FakeLoki(lines)
    start, end = start_end(lines)
    entries, stats = run(
        gh.read_range(loki, PATH, "{job=\"api\"}", gh.parse_rfc3339(start, "s"), gh.parse_rfc3339(end, "e"), 5000)
    )
    assert len(entries) == 128
    assert {n for n, _, _ in entries} == {n for n, _ in lines}
    assert stats.shrunk >= 1
    assert stats.partial is False
    assert stats.page_limit < 128
    # Every request that succeeded stayed under the fake's limit, and after the
    # first refusal no request was larger than the one before it.
    limits = [c["limit"] for c in loki.calls]
    assert limits[0] == gh.lines_per_page(gh.DEFAULT_LINE_BYTES)
    assert limits[1] < limits[0]


def test_proactive_page_size_needs_no_retry():
    # Known line size (query_logs with server-side truncation): no refusals at all.
    lines = window_of(5000, 2000)
    loki = FakeLoki(lines)
    start, end = start_end(lines)
    entries, stats = run(
        gh.read_range(
            loki, PATH, "{job=\"api\"}", gh.parse_rfc3339(start, "s"), gh.parse_rfc3339(end, "e"),
            5000, bytes_per_line=2001 + gh.LINE_OVERHEAD_BYTES,
        )
    )
    assert len(entries) == 5000
    assert stats.shrunk == 0
    assert stats.pages > 1
    assert all(c["limit"] * 2000 <= gh.TARGET_PAGE_BYTES for c in loki.calls)


def test_forward_paging_reads_every_line_once():
    lines = window_of(300, 50_000)
    loki = FakeLoki(lines)
    start, end = start_end(lines)
    entries, stats = run(
        gh.read_range(
            loki, PATH, "{job=\"api\"}", gh.parse_rfc3339(start, "s"), gh.parse_rfc3339(end, "e"),
            1000, direction="forward",
        )
    )
    assert [n for n, _, _ in entries] == [n for n, _ in lines]
    assert stats.pages > 1


def test_single_huge_line_raises_clear_error():
    lines = window_of(20, 5_000_000)
    loki = FakeLoki(lines)
    start, end = start_end(lines)
    with pytest.raises(RuntimeError, match="too large to return"):
        run(gh.read_range(loki, PATH, "{}", gh.parse_rfc3339(start, "s"), gh.parse_rfc3339(end, "e"), 100))
    assert loki.calls[-1]["limit"] == gh.MIN_PAGE_LINES


def test_huge_lines_after_some_pages_return_partial():
    # Newest lines are small, older ones are each over the limit on their own.
    small = [(BASE + 10_000_000 + i, f"s{i}") for i in range(50)]
    huge = [(BASE + i, "h" * 5_000_000) for i in range(30)]
    loki = FakeLoki(huge + small)
    entries, stats = run(
        gh.read_range(
            loki, PATH, "{}", gh.parse_rfc3339(gh.from_nanos(BASE - 1), "s"),
            gh.parse_rfc3339(gh.from_nanos(BASE + 20_000_000), "e"), 5000,
        )
    )
    assert stats.partial is True
    assert sorted(t for _, t, _ in entries) == sorted(t for _, t in small)


# --- the app functions --------------------------------------------------------


class StubClient(FakeLoki):
    async def datasource_path(self, datasource):
        return "/api/datasources/proxy/uid/loki"


def app_with(loki):
    app = App()
    app.client = loki
    import logging

    app.logger = logging.getLogger("test")
    return app


def test_log_digest_reports_shrunk_pages():
    lines = window_of(128, 72_000)
    loki = StubClient(lines)
    start, end = start_end(lines)
    out = run(app_with(loki).log_digest(LogDigestInput(query="{job=\"api\"}", start=start, end=end, limit=5000)))
    assert out.total_lines == 128
    assert out.pages_shrunk >= 1
    assert out.page_limit < 128
    assert out.partial is False
    assert out.truncated is False


def test_query_logs_truncates_in_loki_and_pages():
    # Real Loki would cut lines with the line_format stage; the fake checks it was asked.
    lines = window_of(5000, 150)
    loki = StubClient(lines)
    start, end = start_end(lines)
    out = run(
        app_with(loki).query_logs(
            QueryLogsInput(query="{job=\"api\"}", start=start, end=end, limit=5000, max_line_length=2000)
        )
    )
    assert out.count == 5000
    assert out.pages > 1 and out.pages_shrunk == 0
    assert out.page_limit * (2001 + gh.LINE_OVERHEAD_BYTES) <= gh.TARGET_PAGE_BYTES
    assert 'line_format "{{ __line__ | trunc 2001 }}"' in loki.calls[0]["query"]
    # Newest first, no line repeated across page boundaries.
    stamps = [line.timestamp for line in out.lines]
    assert stamps == sorted(stamps, reverse=True) and len(set(stamps)) == 5000


def test_query_logs_metric_query_is_not_rewritten():
    lines = window_of(5, 10)
    loki = StubClient(lines)
    start, end = start_end(lines)
    query = 'sum(count_over_time({job="api"}[1m]))'
    run(app_with(loki).query_logs(QueryLogsInput(query=query, start=start, end=end)))
    assert loki.calls[0]["query"] == query


def test_query_logs_error_then_success():
    lines = window_of(128, 72_000)
    loki = StubClient(lines)
    start, end = start_end(lines)
    # No server-side cut in the fake, so the first page is refused like in production.
    out = run(app_with(loki).query_logs(QueryLogsInput(query="{job=\"api\"}", start=start, end=end, limit=5000, max_line_length=20000)))
    assert out.count == 128
    assert out.pages_shrunk >= 1
    assert all(len(line.line) <= 20000 for line in out.lines)


# --- log_digest cuts lines in Loki ---------------------------------------------


def test_salvaged_cut_json_keeps_grouping_fields():
    import json

    full = json.dumps(
        {
            "level": "warn",
            "component": "database",
            "time": "2026-11-04T13:38:40.574800188Z",
            "message": "/app/internal/domain/user/repository.go:117 SLOW SQL >= 100ms\n" + "SELECT 1, " * 9000,
            "app_ref": "x",
        }
    )
    cut = full.encode()[: gh.DIGEST_LINE_BYTES + 1].decode("utf-8", "ignore")
    whole, salvaged = gh.parse_record(full), gh.parse_record(cut, cut=True)
    assert salvaged["level"] == "warn" and salvaged["component"] == "database"
    assert gh.normalize_line(gh.record_message(salvaged))[:400] == gh.normalize_line(gh.record_message(whole))[:400]
    # Not cut: a broken line stays plain text.
    assert gh.parse_record(cut) == {"message": cut}


def test_cut_at_escape_still_salvages():
    rec = gh.parse_record('{"level":"error","message":"bad \\', cut=True)
    assert rec == {"level": "error", "message": "bad "}


def test_log_digest_asks_loki_to_cut_lines():
    lines = window_of(20, 100)
    loki = StubClient(lines)
    start, end = start_end(lines)
    run(app_with(loki).log_digest(LogDigestInput(query="{job=\"api\"}", start=start, end=end)))
    assert f'trunc {gh.DIGEST_LINE_BYTES + 1} ' in loki.calls[0]["query"]
    assert loki.calls[0]["limit"] * (gh.DIGEST_LINE_BYTES + 1 + gh.LINE_OVERHEAD_BYTES) <= gh.TARGET_PAGE_BYTES
