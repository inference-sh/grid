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
