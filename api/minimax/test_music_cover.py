"""No-network tests for music-cover preprocess/generation error handling.

The app folder is not an importable package (hyphen in the name) and inference.py
uses a relative import, so load it under a synthetic package, mirroring the
importlib approach used by test_minimax_llm.py.
"""

import asyncio
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_DIR = Path(__file__).resolve().parent / "music-cover"
_PKG = "music_cover_app"


def _load_module():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_DIR)]
        sys.modules[_PKG] = pkg
    # Load the symlinked helper first so `from .minimax_helper import ...` resolves.
    h_spec = importlib.util.spec_from_file_location(f"{_PKG}.minimax_helper", _DIR / "minimax_helper.py")
    h_mod = importlib.util.module_from_spec(h_spec)
    sys.modules[h_spec.name] = h_mod
    h_spec.loader.exec_module(h_mod)

    spec = importlib.util.spec_from_file_location(f"{_PKG}.inference", _DIR / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


inference = _load_module()


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeClient:
    """Serves queued responses for sequential POSTs; no network."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = 0

    async def post(self, url, headers=None, json=None):
        self.posts += 1
        return self._responses.pop(0)

    async def aclose(self):
        pass


def _app(responses, monkeypatch):
    monkeypatch.setenv("MINIMAX_KEY", "dummy")
    app = inference.App()
    asyncio.run(app.setup(None))
    app.client = _FakeClient(responses)
    return app


def _input():
    # model_construct bypasses validation (and File(uri=...) which would try to
    # download). run() only reads .uri / .prompt / .lyrics / .format.value.
    return inference.AppInput.model_construct(
        reference_audio=SimpleNamespace(uri="https://example.com/ref.mp3"),
        prompt="jazz piano",
        lyrics=None,
        format=inference.FormatEnum.mp3,
    )


def test_preprocess_error_body_is_surfaced(monkeypatch):
    """A failure with no base_resp must surface the real body, not mask it.

    This is the reported bug: the error text used to be thrown away and replaced
    with 'No cover_feature_id ... {}'.
    """
    body = {"error": "music API access revoked for this account"}
    app = _app([_Resp(200, body)], monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(app.run(_input()))
    msg = str(exc.value)
    assert "music API access revoked for this account" in msg, msg
    assert "no cover_feature_id" not in msg.lower(), msg


def test_empty_body_raises(monkeypatch):
    """The user's exact scenario: an empty {} preprocess body fails cleanly."""
    app = _app([_Resp(200, {})], monkeypatch)
    with pytest.raises(RuntimeError):
        asyncio.run(app.run(_input()))


def test_nonzero_status_is_reported(monkeypatch):
    body = {"base_resp": {"status_code": 1004, "status_msg": "insufficient balance"}}
    app = _app([_Resp(200, body)], monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(app.run(_input()))
    msg = str(exc.value)
    assert "insufficient balance" in msg
    # The raw body is appended even when status_msg is present.
    assert "body:" in msg


def test_generation_error_body_is_surfaced(monkeypatch):
    """A generation failure with no base_resp must surface the real body too."""
    pre = {"base_resp": {"status_code": 0}, "cover_feature_id": "cf_123",
           "formatted_lyrics": "la la", "audio_duration": 12.0}
    gen = {"error": "generation quota exceeded"}
    app = _app([_Resp(200, pre), _Resp(200, gen)], monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(app.run(_input()))
    assert "generation quota exceeded" in str(exc.value)


def test_happy_path_reads_toplevel_fields(monkeypatch):
    """A valid two-step response must succeed.

    MiniMax returns cover_feature_id at the TOP LEVEL of the preprocess body;
    reading it from a nested "data" object (the shipped code) fails even on
    success, which is the reported bug. This fixture uses the documented shape.
    """
    pre = {"base_resp": {"status_code": 0}, "cover_feature_id": "cf_123",
           "formatted_lyrics": "la la", "audio_duration": 12.0}
    gen = {"base_resp": {"status_code": 0},
           "data": {"audio": "https://example.com/out.mp3", "status": 2},
           "extra_info": {"music_duration": 5000}}

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(b"\x00")
    tmp.close()

    async def fake_download(client, url, output_path):
        return tmp.name

    monkeypatch.setattr(inference, "download_file", fake_download)
    app = _app([_Resp(200, pre), _Resp(200, gen)], monkeypatch)
    out = asyncio.run(app.run(_input()))
    assert out.audio is not None
    assert app.client.posts == 2
    # music_duration (5000 ms) is metered as 5.0 s of output.
    assert out.output_meta.outputs[0].seconds == 5.0
