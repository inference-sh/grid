"""No-network tests for elevenlabs/tts: what `run` asks ElevenLabs for per
model, and the realtime function: what it sends the Text to Dialogue socket,
how a session ends, and what it bills.

ElevenLabs is a scripted fake socket; the caller is a fake platform socket.
"""

import asyncio
import base64
import importlib
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("ELEVENLABS_KEY", "test")
sys.path.insert(0, str(Path(__file__).parent))
el_tts = importlib.import_module("tts.inference")

PCM = b"\x01\x00" * 2400         # 100 ms of 24 kHz PCM
GEORGE = el_tts.get_voice_id("george")


def audio(pcm=PCM):
    return {"audio": base64.b64encode(pcm).decode(), "alignment": None, "normalized_alignment": None}


def text(words, new_turn=False):
    return {"events": {"type": "text", "text": words, "new_turn": new_turn}}


FLUSH = {"events": {"type": "flush"}}
END = {"events": {"type": "end"}}
TURN_DONE = {"is_final_audio_for_turn": True}
FINAL = {"is_final": True}


class FakeEleven:
    """ElevenLabs' dialogue socket: plays a script of messages ("close" ends
    the socket, "hang" goes quiet), answers ``close_socket`` with `on_close`,
    and records what it was sent."""

    def __init__(self, script, on_close=()):
        self.queue = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)
        self.on_close = list(on_close)
        self.sent = []
        self.url = None
        self.headers = None

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
    """The platform socket the kernel hands the app: frames from the caller,
    then the caller closes after `close_after` seconds of nothing left."""

    def __init__(self, frames=(), close_after=30.0):
        self.frames = list(frames)
        self.close_after = close_after
        self.out = []
        self.closed = False
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

    def audio(self):
        return [o for o in self.out if isinstance(o, (bytes, bytearray))]

    def patches(self):
        return [o for o in self.out if isinstance(o, dict) and "$error" not in o]

    def errors(self):
        return [o["$error"]["message"] for o in self.out if isinstance(o, dict) and "$error" in o]


def realtime(script, caller, on_close=(FINAL, "close"), **input_fields):
    """Runs one realtime session; returns (yields, error, eleven)."""
    eleven = FakeEleven(script, on_close)

    async def connect(url, additional_headers=None, **kwargs):
        eleven.url, eleven.headers = url, additional_headers
        return eleven

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = el_tts.App()
        await app.setup()
        yields = []
        try:
            async for out in app.realtime(el_tts.RealtimeInput(**input_fields), caller):
                yields.append(out)
        except Exception as err:  # noqa: BLE001 - the error is the result under test
            return yields, err
        return yields, None

    yields, err = asyncio.run(go())
    return yields, err, eleven


# ------------------------------------------------------------ what ElevenLabs is sent


def test_the_voice_is_registered_first_and_text_follows_as_dialogue_inputs():
    caller = FakeCaller(frames=[text("Hello, "), text("world.", new_turn=True), FLUSH], close_after=0.1)
    _, _, eleven = realtime(["hang"], caller, stability=0.3, language_code="en")

    assert eleven.url == (
        "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?model_id=eleven_v4_turbo&output_format=pcm_24000&language_code=en"
    )
    assert eleven.headers == {"xi-api-key": "test"}
    assert eleven.sent == [
        {"voices": [GEORGE], "voice_settings": {"stability": 0.3}},
        {"inputs": [{"text": "Hello, ", "voice_id": GEORGE, "new_turn": False}]},
        {"inputs": [{"text": "world.", "voice_id": GEORGE, "new_turn": True}]},
        {"flush": True},
        {"close_socket": True},
    ]


def test_a_custom_voice_and_the_quality_model_are_passed_through():
    _, _, eleven = realtime(["hang"], FakeCaller(frames=[text("Hi.")], close_after=0.1), voice_id="custom123", model="eleven_v4")

    assert "model_id=eleven_v4&" in eleven.url
    assert eleven.sent[0]["voices"] == ["custom123"]
    assert eleven.sent[1]["inputs"][0]["voice_id"] == "custom123"


def test_a_quiet_session_is_kept_alive(monkeypatch):
    monkeypatch.setattr(el_tts, "KEEP_ALIVE_SECONDS", 0.5)
    _, _, eleven = realtime(["hang"], FakeCaller(close_after=1.8))

    assert {"keep_alive": True} in eleven.sent


# ------------------------------------------------------------ the speech


def test_speech_reaches_the_caller_as_binary_frames_and_progress_as_patches():
    caller = FakeCaller(frames=[text("Hello there."), FLUSH], close_after=0.3)
    yields, err, _ = realtime([audio(), audio(), TURN_DONE, "hang"], caller)

    assert err is None
    assert caller.audio() == [PCM, PCM]
    assert caller.patches() == [{"characters": 0, "seconds": 0.0}, {"characters": 12, "seconds": 0.2}]
    assert [(y.characters, y.seconds, y.partial) for y in yields] == [(12, 0.2, True), (12, 0.2, False)]


# ------------------------------------------------------------ ending a session


def test_end_lets_elevenlabs_finish_speaking_before_the_result():
    caller = FakeCaller(frames=[text("Goodbye for now."), END])
    yields, err, eleven = realtime(["hang"], caller, on_close=[audio(), audio(), audio(), FINAL, "close"])

    assert err is None
    assert eleven.sent[-1] == {"close_socket": True}
    assert caller.audio() == [PCM] * 3, "the speech still to come is delivered"
    result = yields[-1]
    assert (result.partial, result.end_reason, result.seconds) == (False, "the caller ended the session", 0.3)
    meta = result.output_meta.outputs[0]
    assert (meta.seconds, meta.extra) == (0.3, {"characters": 16, "model": "eleven_v4_turbo"})
    assert result.output_meta.inputs == []


def test_a_caller_that_just_leaves_ends_the_session_at_once():
    caller = FakeCaller(frames=[text("Hello there.")], close_after=0.05)
    yields, err, eleven = realtime(["hang"], caller, on_close=["hang"])

    assert err is None
    assert eleven.sent[-1] == {"close_socket": True}
    assert yields[-1].output_meta.outputs[0].extra["characters"] == 12, "what was sent is still billed"


def test_elevenlabs_closing_a_session_that_spoke_is_a_normal_billed_end():
    caller = FakeCaller(frames=[text("Hello there.")])
    yields, err, _ = realtime([audio(), FINAL, "close"], caller)

    assert err is None
    result = yields[-1]
    assert (result.partial, result.end_reason) == (False, "ElevenLabs closed the session")
    assert result.output_meta.outputs[0].extra["characters"] == 12


def test_a_session_elevenlabs_refuses_fails_the_task():
    refused = {"message": "Voice not found", "error": "voice_not_found", "code": 1008}
    caller = FakeCaller(frames=[text("Hello there.")])
    yields, err, _ = realtime([refused, "close"], caller)

    assert yields == []
    assert isinstance(err, RuntimeError) and "Voice not found" in str(err)
    assert caller.errors() == ["ElevenLabs: Voice not found"]


def test_no_text_for_idle_minutes_ends_the_session_with_a_result():
    caller = FakeCaller(frames=[text("Hello there.")])
    yields, err, _ = realtime([audio(), "hang"], caller, idle_minutes=0.002)

    assert err is None
    assert yields[-1].end_reason == "ended after 0.002 minutes with no text"
    assert yields[-1].output_meta.outputs[0].extra["characters"] == 12


# ------------------------------------------------------------ a whole text


def run(monkeypatch, **input_fields):
    asked = {}

    def fake_tts(**kwargs):
        asked.update(kwargs)
        return "/tmp/speech.mp3"

    monkeypatch.setattr(el_tts, "text_to_speech", fake_tts)
    monkeypatch.setattr(el_tts, "get_audio_duration", lambda path, logger=None: 3.5)

    async def go():
        app = el_tts.App()
        await app.setup()
        return await app.run(el_tts.AppInput(**input_fields))

    try:
        return asyncio.run(go()), None, asked
    except Exception as err:  # noqa: BLE001 - the error is the result under test
        return None, err, asked


def test_v4_is_asked_for_with_its_two_voice_settings_only(monkeypatch):
    out, err, asked = run(monkeypatch, text="[whispers] Hello.", model="eleven_v4", audio_tags=True, stability=0.4, style=0.9)

    assert err is None
    assert asked["model_id"] == "eleven_v4"
    assert asked["voice_settings"] == {"stability": 0.4, "similarity_boost": 0.75}
    assert out.output_meta.outputs[0].extra == {"characters": 17, "model": "eleven_v4"}


def test_older_models_keep_style_and_speaker_boost(monkeypatch):
    _, _, asked = run(monkeypatch, text="Hello.", model="eleven_v3", style=0.2)

    assert asked["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2, "use_speaker_boost": True}


def test_each_model_has_its_own_character_limit(monkeypatch):
    _, too_long_v4, _ = run(monkeypatch, text="a" * 10001, model="eleven_v4")
    ok_v4, no_error, _ = run(monkeypatch, text="a" * 10000, model="eleven_v4")
    _, too_long_v3, _ = run(monkeypatch, text="a" * 5001, model="eleven_v3")

    assert "10000 character limit" in str(too_long_v4) and "5000 character limit" in str(too_long_v3)
    assert no_error is None and ok_v4 is not None


def test_audio_tags_are_refused_on_models_that_would_read_them_aloud(monkeypatch):
    _, err, _ = run(monkeypatch, text="[laughs] Hello.", model="eleven_flash_v2_5", audio_tags=True)

    assert isinstance(err, ValueError) and "eleven_v4 and eleven_v3" in str(err)
