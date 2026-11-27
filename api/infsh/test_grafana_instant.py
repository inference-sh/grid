"""infsh/grafana: query_logs_instant reads Loki's sample times as seconds.

Loki answers a metric query with [<unix seconds as float>, "<value>"] and a
log query with ["<unix nanoseconds>", "<line>"]; reading the first as
nanoseconds put every count at 1970-01-01.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grafana import grafana_http as gh  # noqa: E402
from grafana.inference import App, QueryLogsInstantInput  # noqa: E402

SECONDS = 1764245754.123
NANOS = "1764245754123000000"
WANT = "2025-11-27T12:15:54.123000+00:00"


class Stub:
    def __init__(self, data):
        self.data = data

    async def datasource_path(self, datasource):
        return "/api/datasources/proxy/uid/loki"

    async def request(self, method, path, *, params=None, json_body=None):
        return {"data": self.data}


def instant(data):
    app = App()
    app.client = Stub(data)
    app.logger = logging.getLogger("test")
    return asyncio.run(app.query_logs_instant(QueryLogsInstantInput(query="x"))).results


def test_from_seconds():
    assert gh.from_seconds(SECONDS) == WANT
    assert gh.from_seconds("1764245754.123") == WANT
    assert gh.from_seconds("nope") == "nope"


def test_vector_sample_is_seconds():
    [r] = instant({"resultType": "vector", "result": [{"metric": {}, "value": [SECONDS, "1257"]}]})
    assert (r.timestamp, r.value) == (WANT, "1257")


def test_matrix_samples_are_seconds():
    [r] = instant({"resultType": "matrix", "result": [{"metric": {}, "values": [[SECONDS, "3"]]}]})
    assert r.timestamp == WANT


def test_stream_entries_stay_nanoseconds():
    [r] = instant({"resultType": "streams", "result": [{"stream": {}, "values": [[NANOS, "line"]]}]})
    assert (r.timestamp, r.line) == (WANT, "line")
