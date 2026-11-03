"""Tests for infsh/agent-browser's URL check (native/agents/agent-browser).

Run: python -m pytest native/agents/test_agent_browser.py -q

No browser: playwright and inferencesh are stubbed when they are not
installed, and only the module's own URL functions run.
"""

import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "agent-browser", "inference.py")


def _stub_missing():
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        pw = types.ModuleType("playwright")
        api = types.ModuleType("playwright.async_api")
        api.async_playwright = None
        api.Page = api.Browser = api.BrowserContext = object
        pw.async_api = api
        sys.modules.update({"playwright": pw, "playwright.async_api": api})
    try:
        import inferencesh  # noqa: F401
    except ImportError:
        from pydantic import BaseModel

        inf = types.ModuleType("inferencesh")

        class BaseApp:
            pass

        inf.BaseApp = BaseApp
        inf.BaseAppInput = inf.BaseAppOutput = inf.File = BaseModel
        sys.modules["inferencesh"] = inf


@pytest.fixture(scope="module")
def app():
    _stub_missing()
    spec = importlib.util.spec_from_file_location("agent_browser_inference", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "url, want",
    [
        ("https://example.com/a?b=c", "https://example.com/a?b=c"),
        ("http://example.com", "http://example.com"),
        ("HTTPS://example.com", "HTTPS://example.com"),
        ("example.com", "https://example.com"),
        ("www.example.com/path", "https://www.example.com/path"),
        ("example.com:8080/x", "https://example.com:8080/x"),
        ("localhost:3000", "https://localhost:3000"),
        ("  https://example.com  ", "https://example.com"),
        ("data:text/html,<h1>hi</h1>", "data:text/html,<h1>hi</h1>"),
    ],
)
def test_allowed(app, url, want):
    assert app.normalize_url(url) == want


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "FILE:///etc/passwd",
        "File:/proc/self/environ",
        "file:etc/passwd",
        "  file:///home/inferencesh",
        "javascript:alert(1)",
        "chrome://settings",
        "view-source:file:///etc/passwd",
        "blob:https://example.com/uuid",
        "filesystem:https://example.com/temporary/x",
        "about:blank",
        "",
        "   ",
    ],
)
def test_refused(app, url):
    with pytest.raises(ValueError):
        app.normalize_url(url)


# A URL with no scheme gets https://: whatever Chromium strips from it
# (tabs, newlines), it stays an https URL, never file:.
@pytest.mark.parametrize("url", ["fi\tle:///etc/passwd", "fi\nle:///etc/passwd", "\x00file:///etc/passwd"])
def test_mangled_scheme_stays_https(app, url):
    assert app.normalize_url(url).startswith("https://")


def test_is_local_page(app):
    assert app.is_local_page("file:///etc/passwd")
    assert app.is_local_page("FILE:///x")
    assert not app.is_local_page("https://example.com/file:///x")
    assert not app.is_local_page("about:blank")
    assert not app.is_local_page("")


def test_method_uses_the_check(app):
    with pytest.raises(ValueError):
        app.App._normalize_url(None, "file:///etc/passwd")


# --- uploads ---

from types import SimpleNamespace  # noqa: E402


@pytest.fixture
def upload_dir(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    (d / "ok.txt").write_bytes(b"fine")
    return str(d)


def _no_fetch(uri):
    raise AssertionError(f"fetched {uri}")


@pytest.mark.parametrize(
    "path",
    [
        "/proc/self/environ",
        "/etc/passwd",
        "inference.py",
        "../../etc/passwd",
    ],
)
def test_upload_refuses_container_files(app, upload_dir, path):
    with pytest.raises(ValueError):
        app.upload_payloads([path], None, upload_dir, fetch=_no_fetch)


def test_upload_refuses_escapes_from_the_upload_dir(app, upload_dir, tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("s3cret")
    os.symlink(secret, os.path.join(upload_dir, "link"))
    os.symlink("/proc/self/environ", os.path.join(upload_dir, "env"))
    for p in [os.path.join(upload_dir, "link"), os.path.join(upload_dir, "env"),
              os.path.join(upload_dir, "..", "secret"), upload_dir, os.path.join(upload_dir, "missing")]:
        with pytest.raises(ValueError):
            app.upload_payloads([p], None, upload_dir, fetch=_no_fetch)


def test_upload_from_the_upload_dir(app, upload_dir):
    got = app.upload_payloads([os.path.join(upload_dir, "ok.txt")], None, upload_dir, fetch=_no_fetch)
    assert got == [{"name": "ok.txt", "mimeType": "text/plain", "buffer": b"fine"}]


# A task input File counts by its URI: a local path it carries (one the
# caller may have named) is never read; the URI is downloaded afresh.
def test_upload_input_files_by_uri_only(app, upload_dir, tmp_path):
    fetched = tmp_path / "fetched.png"
    fetched.write_bytes(b"\x89PNG")
    seen = []

    def fetch(uri):
        seen.append(uri)
        return str(fetched)

    f = SimpleNamespace(uri="https://example.com/a/pic.png?x=1", path="/proc/self/environ")
    got = app.upload_payloads(["https://example.com/b.pdf"], [f], upload_dir, fetch=fetch)
    assert seen == ["https://example.com/b.pdf", "https://example.com/a/pic.png?x=1"]
    assert [p["name"] for p in got] == ["b.pdf", "pic.png"]
    assert all(p["buffer"] == b"\x89PNG" for p in got)

    for bad in [SimpleNamespace(uri="/proc/self/environ", path="/proc/self/environ"),
                SimpleNamespace(uri="file:///etc/passwd", path=None),
                SimpleNamespace(uri=None, path="/etc/passwd")]:
        with pytest.raises(ValueError):
            app.upload_payloads(None, [bad], upload_dir, fetch=_no_fetch)


def test_upload_needs_a_file(app, upload_dir):
    with pytest.raises(ValueError):
        app.upload_payloads(None, None, upload_dir, fetch=_no_fetch)
    with pytest.raises(ValueError):
        app.upload_payloads([], [], upload_dir, fetch=_no_fetch)


def test_upload_size_cap(app, upload_dir, monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 3)
    with pytest.raises(ValueError):
        app.upload_payloads([os.path.join(upload_dir, "ok.txt")], None, upload_dir, fetch=_no_fetch)
