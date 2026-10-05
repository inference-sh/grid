"""No-network tests for the per-model ElevenLabs text to speech apps: each one
asks for its own model with the settings that model takes, holds its own
character limit, and reports the characters it is billed for. The two v4 apps
also speak live.

ElevenLabs is a fake: `text_to_speech` for a whole text, a scripted socket for
the live function.
"""

import asyncio
import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("ELEVENLABS_KEY", "test")

APPS = {
    "eleven-v4": ("eleven_v4", 10000, False, True),
    "eleven-v4-turbo": ("eleven_v4_turbo", 10000, False, True),
    "eleven-v3": ("eleven_v3", 5000, True, False),
    "eleven-multilingual-v2": ("eleven_multilingual_v2", 10000, True, False),
    "eleven-flash-v2-5": ("eleven_flash_v2_5", 40000, True, False),
}


def load(name):
    """The app as the platform loads it: a package, so its relative imports resolve."""
    package = name.replace("-", "_")
    if package not in sys.modules:
        folder = Path(__file__).with_name(name)
        spec = importlib.util.spec_from_file_location(package, folder / "__init__.py", submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package}.inference"], sys.modules[f"{package}.elevenlabs_tts"]


def run(name, monkeypatch, tmp_path, audio=b"x" * 100, **input_fields):
    app_module, tts = load(name)
    asked = {}

    def fake_tts(**kwargs):
        asked.update(kwargs)
        path = tmp_path / "speech.bin"
        path.write_bytes(audio)
        return str(path)

    monkeypatch.setattr(tts, "text_to_speech", fake_tts)
    monkeypatch.setattr(tts, "get_audio_duration", lambda path, logger=None: 3.5)

    async def go():
        app = app_module.App()
        await app.setup()
        return await app.run(app_module.AppInput(**input_fields))

    try:
        return asyncio.run(go()), None, asked
    except Exception as err:  # noqa: BLE001 - the error is the result under test
        return None, err, asked


# ------------------------------------------------------------ a whole text


@pytest.mark.parametrize("name", APPS)
def test_each_app_asks_for_its_own_model_and_reports_what_is_billed(name, monkeypatch, tmp_path):
    model, _, styled, _ = APPS[name]
    out, err, asked = run(name, monkeypatch, tmp_path, text="Hello there.", stability=0.4)

    assert err is None
    assert asked["model_id"] == model
    expected = {"stability": 0.4, "similarity_boost": 0.75}
    if styled:
        expected.update(style=0.0, use_speaker_boost=True)
    assert asked["voice_settings"] == expected
    meta = out.output_meta
    assert meta.inputs == []
    assert (meta.outputs[0].seconds, meta.outputs[0].extra) == (3.5, {"characters": 12, "model": model})


@pytest.mark.parametrize("name", APPS)
def test_each_app_holds_its_own_character_limit(name, monkeypatch, tmp_path):
    model, limit, _, _ = APPS[name]
    ok, no_error, _ = run(name, monkeypatch, tmp_path, text="a" * limit)
    _, too_long, asked = run(name, monkeypatch, tmp_path, text="a" * (limit + 1))

    assert no_error is None and ok is not None
    assert isinstance(too_long, ValueError) and f"{limit} character limit of {model}" in str(too_long)


def test_only_the_models_before_v4_take_style_and_speaker_boost():
    for name, (_, _, styled, _) in APPS.items():
        fields = load(name)[0].AppInput.model_fields
        assert ("style" in fields, "use_speaker_boost" in fields) == (styled, styled), name
        assert "model" not in fields, "an app is one model"


def test_a_custom_voice_overrides_the_premade_one(monkeypatch, tmp_path):
    _, _, asked = run("eleven-v4", monkeypatch, tmp_path, text="Hi.", voice="alice", voice_id="custom123")

    assert asked["voice_id"] == "custom123"


def test_raw_pcm_is_timed_by_its_size(monkeypatch, tmp_path):
    out, _, _ = run("eleven-flash-v2-5", monkeypatch, tmp_path, audio=b"\0" * 48000, text="Hi.", output_format="pcm_24000")

    assert out.output_meta.outputs[0].seconds == 1.0


def test_only_the_v4_apps_speak_live():
    for name, (_, _, _, live) in APPS.items():
        assert hasattr(load(name)[0].App, "realtime") == live, name


# ------------------------------------------------------------ a conversation


class FakeHttp:
    """httpx.AsyncClient, answering one POST and recording it."""

    def __init__(self, status=200, content=b"ID3 audio", text=""):
        self.status, self.content, self.text, self.request = status, content, text, None

    def __call__(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, params=None, headers=None, json=None):
        self.request = {"url": url, "params": params, "headers": headers, "json": json}
        return types.SimpleNamespace(status_code=self.status, content=self.content, text=self.text)


def dialogue(name, monkeypatch, http=None, **input_fields):
    app_module, tts = load(name)
    http = http or FakeHttp()
    monkeypatch.setattr(tts.httpx, "AsyncClient", http)
    monkeypatch.setattr(tts, "get_audio_duration", lambda path, logger=None: 4.2)

    async def go():
        app = app_module.App()
        await app.setup()
        return await app.dialogue(tts.DialogueInput(**input_fields))

    try:
        return asyncio.run(go()), None, http
    except Exception as err:  # noqa: BLE001 - the error is the result under test
        return None, err, http


LINES = [
    {"text": "[excited] Did you hear the news?", "voice": "jessica"},
    {"text": "[interrupting] I did!", "voice_id": "custom123"},
]


@pytest.mark.parametrize("name", ["eleven-v4", "eleven-v3"])
def test_a_dialogue_is_one_request_on_the_apps_model_billed_by_its_characters(name, monkeypatch):
    model = APPS[name][0]
    _, tts = load(name)
    out, err, http = dialogue(name, monkeypatch, segments=LINES, language_code="en")

    assert err is None
    assert http.request["url"] == "https://api.elevenlabs.io/v1/text-to-dialogue"
    assert http.request["params"] == {"output_format": "mp3_44100_128"}
    assert http.request["headers"] == {"xi-api-key": "test"}
    assert http.request["json"] == {
        "inputs": [
            {"text": "[excited] Did you hear the news?", "voice_id": tts.get_voice_id("jessica")},
            {"text": "[interrupting] I did!", "voice_id": "custom123"},
        ],
        "model_id": model,
        "language_code": "en",
    }
    with open(out.audio.path, "rb") as audio:
        assert audio.read() == b"ID3 audio"
    meta = out.output_meta.outputs[0]
    assert (meta.seconds, meta.extra) == (4.2, {"characters": 53, "model": model})


def test_only_the_v4_and_v3_apps_have_dialogue():
    with_dialogue = {name for name in APPS if hasattr(load(name)[0].App, "dialogue")}

    assert with_dialogue == {"eleven-v4", "eleven-v3"}, "v4 Turbo speaks dialogue over the socket only; the v2 models have none"


def test_a_dialogue_holds_the_models_character_limit_across_its_lines(monkeypatch):
    lines = [{"text": "a" * 3000, "voice": "george"}, {"text": "b" * 2001, "voice": "aria"}]
    out, err, http = dialogue("eleven-v3", monkeypatch, segments=lines)

    assert out is None and http.request is None, "nothing is sent"
    assert isinstance(err, ValueError) and "5001 characters; eleven_v3 takes 5000" in str(err)


def test_a_dialogue_takes_up_to_ten_voices(monkeypatch):
    lines = [{"text": "Hi.", "voice_id": f"voice{i}"} for i in range(11)]
    _, err, http = dialogue("eleven-v4", monkeypatch, segments=lines)

    assert http.request is None
    assert isinstance(err, ValueError) and "up to 10 different voices; this one has 11" in str(err)


def test_an_elevenlabs_refusal_fails_the_dialogue_with_what_it_said(monkeypatch):
    http = FakeHttp(status=422, text='{"detail": "voice not found"}')
    out, err, _ = dialogue("eleven-v4", monkeypatch, http=http, segments=LINES)

    assert out is None
    assert isinstance(err, RuntimeError) and "422" in str(err) and "voice not found" in str(err)


# ------------------------------------------------------------ the live function


PCM = b"\x01\x00" * 2400         # 100 ms of 24 kHz PCM
FINAL = {"is_final": True}


class FakeEleven:
    """ElevenLabs' dialogue socket: plays a script ("close" ends the socket,
    "hang" goes quiet), answers ``close_socket`` with `on_close`, records what
    it was sent."""

    def __init__(self, script, on_close):
        self.queue = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)
        self.on_close, self.sent, self.url = list(on_close), [], None

    async def send(self, data):
        message = json.loads(data)
        self.sent.append(message)
        if message.get("close_socket"):
            for item in self.on_close:
                self.queue.put_nowait(item)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            item = await self.queue.get()
            await asyncio.sleep(0.01)
            if item == "close":
                raise StopAsyncIteration
            if item == "hang":
                continue
            return json.dumps(item)

    async def close(self):
        pass


class FakeCaller:
    """The platform socket: frames from the caller, then nothing until `close_after`."""

    def __init__(self, frames=(), close_after=30.0):
        self.frames, self.close_after, self.out, self.closed = list(frames), close_after, [], False
        self.id, self.metadata, self.dropped, self.binary_backlog = "s", {}, 0, 256

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for frame in self.frames:
            await asyncio.sleep(0.02)
            yield json.dumps(frame)
        await asyncio.sleep(self.close_after)

    async def send(self, data):
        self.out.append(data)

    async def close(self):
        self.closed = True


def realtime(name, script, caller, on_close=(FINAL, "close"), **input_fields):
    app_module, tts = load(name)
    eleven = FakeEleven(script, on_close)

    async def connect(url, **kwargs):
        eleven.url = url
        return eleven

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = app_module.App()
        await app.setup()
        return [out async for out in app.realtime(tts.RealtimeInput(**input_fields), caller)]

    return asyncio.run(go()), eleven


@pytest.mark.parametrize("name", ["eleven-v4", "eleven-v4-turbo"])
def test_a_live_session_speaks_on_the_apps_model_and_bills_the_characters(name):
    model = APPS[name][0]
    speech = {"audio": base64.b64encode(PCM).decode()}
    caller = FakeCaller(frames=[{"events": {"type": "text", "text": "Goodbye for now."}}, {"events": {"type": "end"}}])
    yields, eleven = realtime(name, ["hang"], caller, on_close=[speech, speech, FINAL, "close"])

    assert f"model_id={model}&output_format=pcm_24000" in eleven.url
    assert [m for m in eleven.sent if "inputs" in m][0]["inputs"][0]["text"] == "Goodbye for now."
    assert [o for o in caller.out if isinstance(o, bytes)] == [PCM, PCM], "the speech still to come after `end` is delivered"
    result = yields[-1]
    assert (result.partial, result.end_reason, result.seconds) == (False, "the caller ended the session", 0.2)
    meta = result.output_meta.outputs[0]
    assert (meta.seconds, meta.extra) == (0.2, {"characters": 16, "model": model})
