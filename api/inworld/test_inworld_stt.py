"""No-network tests for the realtime function of inworld/speech-to-text: the
transcript Inworld's messages add up to, how a live session ends and is
billed, and what the app asks Inworld for.

Inworld is a scripted fake socket; the caller is a fake platform socket.
"""

import asyncio
import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("INWORLD_KEY", "dGVzdA==")

_dir = Path(__file__).with_name("speech-to-text")
_spec = importlib.util.spec_from_file_location("inworld_stt", _dir / "__init__.py", submodule_search_locations=[str(_dir)])
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["inworld_stt"] = _pkg
_spec.loader.exec_module(_pkg)
inworld_stt = sys.modules["inworld_stt.inference"]

FRAME = b"\x01\x00" * 320        # 20 ms of 16 kHz PCM
USAGE = {"result": {"usage": {"transcribedAudioMs": 6430, "modelId": "inworld/inworld-stt-1"}}}


def heard(text, final=False, words=None):
    transcription = {"transcript": text, "isFinal": final}
    if words is not None:
        transcription["wordTimestamps"] = words
    return {"result": {"transcription": transcription}}


class FakeInworld:
    """Inworld's streaming socket: plays a script of messages ("close" ends
    the socket, "hang" goes quiet), answers ``closeStream`` with `on_close`,
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
        if "closeStream" in message:
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

    def audio(self):
        return [base64.b64decode(m["audioChunk"]["content"]) for m in self.sent if "audioChunk" in m]


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
            yield frame
        await asyncio.sleep(self.close_after)

    async def send(self, data):
        self.out.append(data)

    async def close(self):
        self.closed = True

    def texts(self):
        return [o["text"] for o in self.out if isinstance(o, dict) and "text" in o]

    def errors(self):
        return [o["$error"]["message"] for o in self.out if isinstance(o, dict) and "$error" in o]


def realtime(script, caller, on_close=("close",), **input_fields):
    """Runs one realtime session; returns (yields, error, inworld)."""
    inworld = FakeInworld(script, on_close)

    async def connect(url, additional_headers=None, **kwargs):
        inworld.url, inworld.headers = url, additional_headers
        return inworld

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = inworld_stt.App()
        await app.setup()
        yields = []
        try:
            async for out in app.realtime(inworld_stt.RealtimeInput(**input_fields), caller):
                yields.append(out)
        except Exception as err:  # noqa: BLE001 - the error is the result under test
            return yields, err
        return yields, None

    yields, err = asyncio.run(go())
    return yields, err, inworld


# ------------------------------------------------------------ the transcript


def test_an_interim_transcript_is_replaced_and_a_final_one_is_kept():
    words = [{"word": "What's", "startTimeMs": 400, "endTimeMs": 700}, {"word": "weather?", "startTimeMs": 900, "endTimeMs": 1900}]
    script = [
        {"result": {"speechStarted": {"startTimeMs": 0, "confidence": 0}}},
        heard("what's"), heard("what's the wea"), heard("What's the weather?", final=True, words=words),
        {"result": {"speechStopped": {"silenceDurationMs": 150}}},
        heard("and to"), "hang",
    ]
    caller = FakeCaller(close_after=0.4)
    yields, err, _ = realtime(script, caller)

    assert err is None
    assert caller.texts() == ["", "what's", "what's the wea", "What's the weather?", "What's the weather? and to"]
    result = yields[-1]
    assert [t.text for t in result.turns] == ["What's the weather?", "and to"], "the turn in progress is kept"
    assert (result.turns[0].start, result.turns[0].end) == (0.4, 1.9)


def test_snake_case_payloads_are_read_the_same():
    script = [{"result": {"transcription": {"transcript": "Hello.", "is_final": True}}}, "hang"]
    on_close = [{"result": {"usage": {"transcribed_audio_ms": 2400}}}, "close"]
    yields, _, _ = realtime(script, FakeCaller(close_after=0.2), on_close=on_close)

    assert yields[-1].text == "Hello."
    assert yields[-1].output_meta.inputs[0].seconds == 2.4


def test_every_finished_turn_is_a_snapshot_and_the_result_is_last():
    yields, _, _ = realtime([heard("One.", final=True), heard("Two.", final=True), "hang"], FakeCaller(close_after=0.3))

    assert [(y.text, y.partial) for y in yields] == [("One.", True), ("One. Two.", True), ("One. Two.", False)]


# ------------------------------------------------------------ ending a session


def test_the_caller_closing_closes_the_stream_and_bills_what_inworld_counted():
    caller = FakeCaller(frames=[FRAME] * 7, close_after=0.1)
    on_close = [heard("Goodbye.", final=True), USAGE, "close"]
    yields, err, inworld = realtime([heard("good"), "hang"], caller, on_close=on_close)

    assert err is None
    assert inworld.sent[-1] == {"closeStream": {}}
    assert b"".join(inworld.audio()) == FRAME * 7, "the frames left over from the last chunk are sent before the close"
    result = yields[-1]
    assert (result.text, result.partial, result.end_reason) == ("Goodbye.", False, "the caller closed the session")
    meta = result.output_meta.inputs[0]
    assert meta.seconds == 6.43, "Inworld's own count of the audio is what is billed"
    assert meta.extra == {"model": "inworld/inworld-stt-1", "caller_seconds": 0.14}


def test_inworld_not_answering_the_close_still_ends_with_a_billed_result(monkeypatch):
    monkeypatch.setattr(inworld_stt, "FLUSH_SECONDS", 0.2)
    caller = FakeCaller(frames=[FRAME] * 50, close_after=0.1)
    yields, err, _ = realtime([heard("Hello.", final=True), heard("and"), "hang"], caller, on_close=[])

    assert err is None
    assert yields[-1].text == "Hello. and"
    assert yields[-1].output_meta.inputs[0].seconds == 1.0, "without Inworld's count, the audio sent is billed"


def test_inworld_closing_a_stream_it_answered_is_a_normal_billed_end():
    yields, err, _ = realtime([heard("Hello.", final=True), "close"], FakeCaller(frames=[FRAME] * 10))

    assert err is None
    result = yields[-1]
    assert (result.partial, result.end_reason, result.text) == (False, "Inworld closed the stream", "Hello.")
    assert result.output_meta.inputs[0].extra["model"] == "inworld/inworld-stt-1"


def test_a_stream_inworld_refuses_fails_the_task():
    refused = {"error": {"code": 3, "message": "invalid transcribe config: unsupported audio encoding"}}
    yields, err, _ = realtime([refused, "close"], FakeCaller())

    assert yields == []
    assert isinstance(err, RuntimeError) and "unsupported audio encoding" in str(err)


def test_a_config_rejected_without_a_word_fails_the_task():
    yields, err, _ = realtime(["close"], FakeCaller())

    assert yields == []
    assert isinstance(err, RuntimeError) and "Inworld closed the stream" in str(err)


def test_nobody_speaking_ends_the_session_with_a_result():
    caller = FakeCaller(frames=[FRAME] * 5)
    yields, err, _ = realtime(["hang"], caller, idle_minutes=0.002)

    assert err is None
    assert yields[-1].end_reason == "ended after 0.002 minutes with nobody speaking"
    assert caller.errors() == ["ended after 0.002 minutes with nobody speaking"]


# ------------------------------------------------------------ what Inworld is sent


def test_the_config_is_the_first_frame_and_the_key_goes_in_a_header():
    _, _, inworld = realtime(["hang"], FakeCaller(close_after=0.05), language="en", keyterms=["Inworld"], silence_ms=800)

    assert inworld.url == "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"
    assert inworld.headers == {"Authorization": "Basic dGVzdA=="}
    assert inworld.sent[0] == {
        "transcribeConfig": {
            "modelId": "inworld/inworld-stt-1", "audioEncoding": "LINEAR16", "sampleRateHertz": 16000,
            "numberOfChannels": 1, "includeWordTimestamps": True, "language": "en", "prompts": ["Inworld"],
            "inworldSttV1Config": {"maxTurnSilence": 800},
        }
    }


def test_turn_detection_is_left_to_inworld_by_default():
    _, _, inworld = realtime(["hang"], FakeCaller(close_after=0.05))

    assert "inworldSttV1Config" not in inworld.sent[0]["transcribeConfig"]


def test_audio_is_sent_in_tenth_of_a_second_chunks():
    frames = [bytes([i, 0]) * 320 for i in range(1, 12)]        # 11 frames of 20 ms
    _, _, inworld = realtime(["hang"], FakeCaller(frames=frames, close_after=0.05))

    assert [len(c) for c in inworld.audio()] == [3200, 3200, 640]
    assert b"".join(inworld.audio()) == b"".join(frames)


def test_a_pause_is_sent_as_silence(monkeypatch):
    monkeypatch.setattr(inworld_stt, "GAP_SECONDS", 0.1)
    yields, _, inworld = realtime(["hang"], FakeCaller(frames=[FRAME], close_after=0.5))

    silence = [a for a in inworld.audio() if a == inworld_stt.SILENCE]
    assert 2 <= len(silence) <= 5, "100 ms of silence per 100 ms of pause, after the gap"
    assert inworld.sent[-1] == {"closeStream": {}}, "none after the caller closed"
    assert yields[-1].seconds == round((len(FRAME) + len(silence) * 3200) / 32000, 3)
