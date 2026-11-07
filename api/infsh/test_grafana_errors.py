"""infsh/grafana: errors from in front of Grafana are told apart from Grafana's own.

Seen in production (annotation 38805): Cloudflare's WAF 403 on a LogQL filter
was reported as "Grafana rejected the credentials", and Grafana's 502 after its
30s dataproxy timeout (body: Cloudflare's custom 5xx page) was retried four
times, 30s each.
"""

import asyncio
import os
import sys
import types

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grafana import grafana_http as gh  # noqa: E402

URL = "https://telemetry.example/api/datasources/proxy/uid/loki/loki/api/v1/query_range"
CF_BLOCK = "<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head></html>"
OFFLINE = "<!doctype html><html><head><title>inference.sh is sleeping</title></head></html>"


def response(status, text="", headers=None):
    return httpx.Response(status, text=text, headers=headers or {}, request=httpx.Request("GET", URL))


def test_edge_block_is_not_a_credential_error():
    with pytest.raises(RuntimeError) as exc:
        gh._result(response(403, CF_BLOCK, {"cf-ray": "a56e-FRA", "server": "cloudflare"}), "GET", URL)
    message = str(exc.value)
    assert "Cloudflare blocked" in message and "a56e-FRA" in message
    assert "credentials were not checked" in message


def test_grafana_403_names_the_permission():
    body = '{"message":"Access denied to datasource","messageId":"datasources.accessDenied"}'
    with pytest.raises(RuntimeError) as exc:
        gh._result(response(403, body, {"cf-ray": "x", "content-type": "application/json"}), "GET", URL)
    message = str(exc.value)
    assert "Access denied to datasource" in message and "datasources:query" in message
    assert "Cloudflare" not in message


def test_grafana_401_points_at_the_token():
    with pytest.raises(RuntimeError) as exc:
        gh._result(response(401, '{"message":"invalid API key"}'), "GET", URL)
    assert "invalid API key" in str(exc.value) and "GRAFANA_API_KEY" in str(exc.value)


def test_html_error_page_is_summarized_not_dumped():
    with pytest.raises(RuntimeError) as exc:
        gh._result(response(502, OFFLINE), "GET", URL)
    message = str(exc.value)
    assert "inference.sh is sleeping" in message and "<html" not in message


def test_slow_502_is_not_retried(monkeypatch):
    monkeypatch.setenv("GRAFANA_URL", "https://telemetry.example")
    monkeypatch.setenv("GRAFANA_API_KEY", "k")
    readings = [0.0]  # first reading 0, every later one 30: the attempt took 30s

    def monotonic():
        value = readings[0]
        readings[0] = 30.0
        return value

    # Patched on the module, not on `time`, so asyncio's own clock is left alone.
    monkeypatch.setattr(gh, "time", types.SimpleNamespace(monotonic=monotonic))
    calls = []

    async def fake_request(method, url, **kwargs):
        calls.append(url)
        return response(502, OFFLINE)

    client = gh.GrafanaClient()
    client._client.request = fake_request
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.request("GET", "/api/x"))
    assert len(calls) == 1
    assert "dataproxy timeout" in str(exc.value)


def test_fast_502_is_retried(monkeypatch):
    monkeypatch.setenv("GRAFANA_URL", "https://telemetry.example")
    monkeypatch.setenv("GRAFANA_API_KEY", "k")
    monkeypatch.setattr(gh, "time", types.SimpleNamespace(monotonic=lambda: 0.0))
    monkeypatch.setattr(gh, "_retry_after", lambda response, attempt: 0.0)
    answers = [response(502), response(200, '{"ok":true}')]

    async def fake_request(method, url, **kwargs):
        return answers.pop(0)

    client = gh.GrafanaClient()
    client._client.request = fake_request
    assert asyncio.run(client.request("GET", "/api/x")) == {"ok": True}
