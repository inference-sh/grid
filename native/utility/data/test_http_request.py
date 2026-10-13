"""Tests for infsh/http (native/utility/data/http-request).

Run: python -m pytest native/utility/data/test_http_request.py -q

A local server on 127.0.0.1 plays the internet. It is reached through the
name public.test, which the tests let past the guard; every other name goes
through the real check, so localhost, a stub name resolving to 127.0.0.1 and
redirects to internal addresses are all refused as they would be in production.
"""

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "http-request")


def _load():
    spec = importlib.util.spec_from_file_location(
        "infsh_http_app", os.path.join(APP_DIR, "__init__.py"), submodule_search_locations=[APP_DIR]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["infsh_http_app"] = pkg
    spec.loader.exec_module(pkg)
    return sys.modules["infsh_http_app.inference"], sys.modules["infsh_http_app.safehttp"]


inference, safehttp = _load()
App, AppInput = inference.App, inference.AppInput
RequestError = inference.RequestError


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype=None, extra=None):
        self.send_response(code)
        if ctype:
            self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _route(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else b""
        p = u.path
        if p == "/echo":
            body = json.dumps({
                "method": self.command,
                "query": parse_qs(u.query),
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": data.decode("latin-1"),
            }).encode()
            return self._send(200, body, "application/json")
        if p == "/setcookie":
            return self._send(200, b"{}", "application/json", {"Set-Cookie": "sid=leak; Path=/"})
        if p == "/json":
            return self._send(200, b'{"a": 1, "b": [true, null]}', "application/json; charset=utf-8")
        if p == "/text":
            return self._send(200, "héllo".encode(), "text/plain; charset=utf-8")
        if p == "/bin":
            return self._send(200, bytes(range(256)) * 4, "image/png",
                              {"Content-Disposition": 'attachment; filename="../../x.png"'})
        if p == "/big":
            return self._send(200, b"x" * int(q["n"]), "text/plain")
        if p == "/stream-big":  # no Content-Length: the cap must hold while streaming
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for _ in range(int(q["chunks"])):
                    self.wfile.write(b"y" * 65536)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client stopped reading at its cap
            return
        if p == "/slow":
            time.sleep(float(q.get("s", 3)))
            return self._send(200, b"late", "text/plain")
        if p == "/status":
            return self._send(int(q["code"]), b'{"error": "nope"}', "application/json")
        if p == "/redirect":
            return self._send(int(q.get("code", 302)), b"", None, {"Location": q["to"]})
        if p == "/loop":
            return self._send(302, b"", None, {"Location": "/loop"})
        if p == "/notjson":
            return self._send(200, b"<html>", "text/html")
        return self._send(404, b"not found", "text/plain")

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _route


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


STUB_DNS = {
    "evil.test": ["127.0.0.1"],
    "metadata.test": ["169.254.169.254"],
    "mixed.test": ["93.184.216.34", "10.0.0.5"],
    "mapped.test": ["::ffff:127.0.0.1"],
}


@pytest.fixture(autouse=True)
def stub_dns(monkeypatch):
    real = safehttp.resolve_checked
    real_resolver = safehttp.resolver

    def resolver(host, port):
        if host in STUB_DNS:
            return STUB_DNS[host]
        return real_resolver(host, port)

    def resolve_checked(host, port):
        if host == "public.test":
            return ["127.0.0.1"]
        return real(host, port)

    monkeypatch.setattr(safehttp, "resolver", resolver)
    monkeypatch.setattr(safehttp, "resolve_checked", resolve_checked)


def run(payload):
    async def go():
        app = App()
        await app.setup()
        try:
            return await app.run(AppInput(**payload))
        finally:
            await app.unload()

    return asyncio.run(go())


def url(port, path):
    return f"http://public.test:{port}{path}"


# --- address guard --------------------------------------------------------

BLOCKED = [
    "127.0.0.1", "127.255.255.254", "10.0.0.1", "172.16.0.1", "172.31.255.255", "192.168.1.1",
    "169.254.169.254", "169.254.0.1", "0.0.0.0", "0.1.2.3", "100.64.0.1", "100.100.100.200",
    "100.127.255.254", "192.0.0.1", "192.0.2.1", "198.18.0.1", "198.19.255.254", "198.51.100.1",
    "203.0.113.1", "240.0.0.1", "255.255.255.255", "224.0.0.1", "239.255.255.250",
    "::1", "::", "fe80::1", "fe80::1%eth0", "fc00::1", "fd00:ec2::254", "ff02::1",
    "64:ff9b::7f00:1", "64:ff9b:1::1", "2001::1", "2001:0:4136:e378::1", "2002:7f00:1::1",
    "2001:db8::1", "::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:10.0.0.1", "::127.0.0.1",
    "fec0::1",
]
ALLOWED = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111", "2001:4860:4860::8888",
           "::ffff:8.8.8.8", "100.63.255.255", "100.128.0.1", "172.32.0.1", "11.0.0.1"]


@pytest.mark.parametrize("ip", BLOCKED)
def test_check_ip_blocks(ip):
    with pytest.raises(safehttp.BlockedAddressError):
        safehttp.check_ip(ip)


@pytest.mark.parametrize("ip", ALLOWED)
def test_check_ip_allows_public(ip):
    safehttp.check_ip(ip)


@pytest.mark.parametrize("target", [
    "http://localhost/", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://evil.test/", "http://metadata.test/",
    "http://mixed.test/", "http://mapped.test/", "http://2130706433/", "http://0x7f.1/", "http://0/",
    "http://[fd00:ec2::254]/", "http://100.100.100.200/",
])
def test_request_to_internal_is_blocked(target):
    with pytest.raises(RequestError, match="blocked destination"):
        run({"url": target})


def test_redirect_to_internal_is_blocked(server):
    for to in ("http://127.0.0.1:%d/json" % server, "http://evil.test/", "http://169.254.169.254/"):
        with pytest.raises(RequestError, match="blocked destination"):
            run({"url": url(server, f"/redirect?to={to}")})


def test_redirect_to_other_scheme_is_blocked(server):
    with pytest.raises(RequestError, match="scheme"):
        run({"url": url(server, "/redirect?to=file:///etc/passwd")})


def test_only_http_schemes():
    for u in ("file:///etc/passwd", "ftp://example.com/", "gopher://x/", "javascript:alert(1)"):
        with pytest.raises(RequestError, match="only http and https"):
            run({"url": u})


def test_unix_socket_refused():
    with pytest.raises(safehttp.BlockedAddressError):
        asyncio.run(safehttp.GuardedAsyncBackend().connect_unix_socket("/var/run/docker.sock"))


# --- requests -------------------------------------------------------------

def test_get_json_auto(server):
    out = run({"url": url(server, "/json")})
    assert out.status == 200 and out.ok
    assert out.body_json == {"a": 1, "b": [True, None]}
    assert out.body_text is None and out.body_file is None
    assert out.headers["content-type"].startswith("application/json")
    assert out.elapsed_ms >= 0


def test_query_merge_and_headers(server):
    out = run({
        "url": url(server, "/echo?x=1"),
        "query": {"y": "2", "z": [3, True]},
        "headers": {"X-Api-Key": "k", "Host": "evil.example", "Connection": "x-drop", "X-Drop": "1",
                    "Transfer-Encoding": "chunked"},
    })
    e = out.body_json
    assert e["query"] == {"x": ["1"], "y": ["2"], "z": ["3", "true"]}
    assert e["headers"]["x-api-key"] == "k"
    assert e["headers"]["host"].startswith("public.test")
    assert "x-drop" not in e["headers"] and "transfer-encoding" not in e["headers"]
    assert e["headers"]["user-agent"].startswith("inference.sh-http")


def test_secret_header_is_filled_from_the_task_environment(server, monkeypatch):
    monkeypatch.setenv("HTTP_SECRET_AUTH_0", "Bearer sk-live-123")
    out = run({
        "url": url(server, "/echo"),
        "headers": {"Authorization": "{{secret.HTTP_SECRET_AUTH_0}}", "X-Mixed": "k={{secret.HTTP_SECRET_AUTH_0}};v=1"},
    })
    h = out.body_json["headers"]
    assert h["authorization"] == "Bearer sk-live-123"
    assert h["x-mixed"] == "k=Bearer sk-live-123;v=1"


def test_secret_header_reads_only_http_secret_names(server, monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "platform-token")
    with pytest.raises(RequestError, match="only HTTP_SECRET_"):
        run({"url": url(server, "/echo"), "headers": {"X-Leak": "{{secret.INFERENCE_API_KEY}}"}})


def test_secret_header_that_was_not_given_is_refused(server, monkeypatch):
    monkeypatch.delenv("HTTP_SECRET_MISSING", raising=False)
    with pytest.raises(RequestError, match="no such secret"):
        run({"url": url(server, "/echo"), "headers": {"Authorization": "{{secret.HTTP_SECRET_MISSING}}"}})


def test_secret_with_a_line_break_is_refused(server, monkeypatch):
    monkeypatch.setenv("HTTP_SECRET_BAD", "a\r\nX-Injected: 1")
    with pytest.raises(RequestError, match="line break or NUL"):
        run({"url": url(server, "/echo"), "headers": {"Authorization": "{{secret.HTTP_SECRET_BAD}}"}})


def test_post_json(server):
    out = run({"method": "POST", "url": url(server, "/echo"), "body_json": {"k": [1, "two"]}})
    e = out.body_json
    assert e["method"] == "POST" and json.loads(e["body"]) == {"k": [1, "two"]}
    assert e["headers"]["content-type"] == "application/json"


def test_post_text_and_form(server):
    e = run({"method": "PUT", "url": url(server, "/echo"), "body_text": "raw",
             "headers": {"Content-Type": "text/csv"}}).body_json
    assert e["body"] == "raw" and e["headers"]["content-type"] == "text/csv"
    e = run({"method": "POST", "url": url(server, "/echo"), "form": {"a": "1", "b": True}}).body_json
    assert e["body"] == "a=1&b=true"
    assert e["headers"]["content-type"] == "application/x-www-form-urlencoded"


def test_multipart_upload(server, tmp_path, monkeypatch):
    monkeypatch.setenv("INFSH_HTTP_ALLOW_LOCAL_FILES", "1")
    f = tmp_path / "hello.txt"
    f.write_bytes(b"file-content-123")
    out = run({
        "method": "POST", "url": url(server, "/echo"), "form": {"note": "hi"},
        "files": [{"field": "upload", "file": {"path": str(f)}, "content_type": "text/plain"}],
    })
    e = out.body_json
    assert e["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    assert 'name="upload"; filename="hello.txt"' in e["body"]
    assert "file-content-123" in e["body"]
    assert 'name="note"' in e["body"] and "hi" in e["body"]


def test_input_file_local_path_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("INFSH_HTTP_ALLOW_LOCAL_FILES", raising=False)
    with pytest.raises(Exception, match="platform file"):
        AppInput(url="https://example.com", files=[{"file": "/etc/passwd"}])


def test_input_file_internal_url_refused():
    with pytest.raises(Exception, match="blocked destination"):
        AppInput(url="https://example.com", files=[{"file": "http://169.254.169.254/latest/meta-data/"}])
    with pytest.raises(Exception, match="blocked destination"):
        AppInput(url="https://example.com", files=[{"file": "http://evil.test/x"}])


def test_input_file_data_uri(server):
    out = run({"method": "POST", "url": url(server, "/echo"),
               "files": [{"file": "data:text/plain;base64,aGk=", "filename": "a.txt"}]})
    assert 'filename="a.txt"' in out.body_json["body"]


def test_one_body_only():
    with pytest.raises(Exception, match="at most one request body"):
        AppInput(url="https://example.com", body_json={"a": 1}, body_text="x")


def test_non_2xx_is_a_result(server):
    out = run({"url": url(server, "/status?code=404")})
    assert out.status == 404 and not out.ok and out.body_json == {"error": "nope"}
    out = run({"url": url(server, "/status?code=503"), "response_as": "json"})
    assert out.status == 503 and out.body_json == {"error": "nope"}


def test_head(server):
    out = run({"method": "HEAD", "url": url(server, "/json")})
    assert out.status == 200 and out.size_bytes == 0 and out.body_json is None


# --- redirects --------------------------------------------------------------

def test_redirect_followed(server):
    out = run({"url": url(server, "/redirect?to=/json")})
    assert out.body_json["a"] == 1 and out.redirects == 1 and out.final_url.endswith("/json")


def test_redirect_not_followed(server):
    out = run({"url": url(server, "/redirect?to=/json"), "follow_redirects": False})
    assert out.status == 302 and out.headers["location"] == "/json" and out.redirects == 0


def test_redirect_limit(server):
    with pytest.raises(RequestError, match="stopped after 3 redirects"):
        run({"url": url(server, "/loop"), "max_redirects": 3})


def test_303_turns_post_into_get(server):
    out = run({"method": "POST", "url": url(server, "/redirect?code=303&to=/echo"), "body_json": {"a": 1}})
    assert out.body_json["method"] == "GET" and out.body_json["body"] == ""


def test_307_resends_body_same_origin(server):
    out = run({"method": "POST", "url": url(server, "/redirect?code=307&to=/echo"), "body_text": "again"})
    assert out.body_json["method"] == "POST" and out.body_json["body"] == "again"


def test_cross_origin_redirect_strips_credentials(server, monkeypatch):
    # other.test is a second origin on the same local server.
    orig = safehttp.resolve_checked

    def rc(host, port):
        return ["127.0.0.1"] if host == "other.test" else orig(host, port)

    monkeypatch.setattr(safehttp, "resolve_checked", rc)
    to = f"http://other.test:{server}/echo"
    out = run({"url": url(server, f"/redirect?to={to}"),
               "headers": {"Authorization": "Bearer s", "X-Api-Key": "k", "Accept": "application/json"}})
    h = out.body_json["headers"]
    assert "authorization" not in h and "x-api-key" not in h and h["accept"] == "application/json"
    with pytest.raises(RequestError, match="another origin"):
        run({"method": "POST", "url": url(server, f"/redirect?code=307&to={to}"), "body_text": "secret"})


# --- size cap, timeout --------------------------------------------------------

def test_size_cap_declared_length(server):
    with pytest.raises(RequestError, match="max_response_bytes"):
        run({"url": url(server, "/big?n=5000"), "max_response_bytes": 1000})


def test_size_cap_streaming(server):
    with pytest.raises(RequestError, match="larger than max_response_bytes"):
        run({"url": url(server, "/stream-big?chunks=64"), "max_response_bytes": 100000})


def test_size_cap_truncate(server):
    out = run({"url": url(server, "/stream-big?chunks=64"), "max_response_bytes": 100000, "truncate": True})
    assert out.truncated and out.size_bytes == 100000
    assert os.path.getsize(out.body_file.path) == 100000


def test_max_response_bytes_hard_cap():
    with pytest.raises(Exception):
        AppInput(url="https://example.com", max_response_bytes=101 * 1024 * 1024)
    with pytest.raises(Exception):
        AppInput(url="https://example.com", timeout_seconds=121)


def test_timeout(server):
    t0 = time.monotonic()
    with pytest.raises(RequestError, match="timed out"):
        run({"url": url(server, "/slow?s=3"), "timeout_seconds": 1})
    assert time.monotonic() - t0 < 2.5


# --- response_as ---------------------------------------------------------------

def test_auto_text(server):
    out = run({"url": url(server, "/text")})
    assert out.body_text == "héllo" and out.body_json is None and out.body_file is None


def test_auto_binary_to_file(server):
    out = run({"url": url(server, "/bin")})
    assert out.body_file is not None and out.body_text is None
    assert os.path.basename(out.body_file.path) == "x.png"  # path traversal in the name is dropped
    with open(out.body_file.path, "rb") as f:
        assert f.read() == bytes(range(256)) * 4


def test_auto_large_text_to_file(server):
    out = run({"url": url(server, f"/big?n={2 * 1024 * 1024}")})
    assert out.body_file is not None and out.body_text is None and out.size_bytes == 2 * 1024 * 1024


def test_response_as_text_json_file(server):
    out = run({"url": url(server, "/json"), "response_as": "text"})
    assert out.body_text.startswith('{"a"') and out.body_json is None
    out = run({"url": url(server, "/json"), "response_as": "file"})
    assert out.body_file is not None and out.body_file.path.endswith(".json")
    out = run({"url": url(server, "/echo"), "response_as": "json"})
    assert out.body_json["method"] == "GET"
    with pytest.raises(RequestError, match="not JSON"):
        run({"url": url(server, "/notjson"), "response_as": "json"})


def test_cookies_do_not_persist_across_requests(server):
    async def go():
        app = App()
        await app.setup()
        try:
            await app.run(AppInput(url=url(server, "/setcookie")))
            return await app.run(AppInput(url=url(server, "/echo")))
        finally:
            await app.unload()

    out = asyncio.run(go())
    assert "cookie" not in out.body_json["headers"]
