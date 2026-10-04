"""No-network tests for the realtime function of elevenlabs/stt: the transcript
ElevenLabs' events add up to, how a live session ends and is billed, and what
the app asks ElevenLabs for.

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
el_stt = importlib.import_module("stt.inference")

READY = {"message_type": "session_started", "session_id": "s", "config": {}}
FRAME = b"\x01\x00" * 320        # 20 ms of 16 kHz PCM


def partial(text):
    return {"message_type": "partial_transcript", "text": text}


def committed(text):
    return {"message_type": "committed_transcript", "text": text}


def timed(text, start, end, language="en"):
    words = [
        {"text": "a", "start": start, "end": start + 0.1, "type": "word"},
        {"text": " ", "start": start + 0.1, "end": end + 9, "type": "spacing"},
        {"text": "b", "start": end - 0.1, "end": end, "type": "word"},
    ]
    return {"message_type": "committed_transcript_with_timestamps", "text": text, "language_code": language, "words": words}


class FakeEleven:
    """ElevenLabs' realtime socket: plays a script of events ("close" ends the
    socket, "hang" goes quiet), answers a commit with `on_commit`, and records
    what it was sent."""

    def __init__(self, script, on_commit=()):
        self.queue = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)
        self.on_commit = list(on_commit)
        self.sent = []
        self.url = None
        self.headers = None

    async def send(self, data):
        message = json.loads(data)
        self.sent.append(message)
        if message.get("commit"):
            for item in self.on_commit:
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
        return [base64.b64decode(m["audio_base_64"]) for m in self.sent]


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


def realtime(script, caller, on_commit=(committed(""), timed("", 0, 0)), **input_fields):
    """Runs one realtime session; returns (yields, error, eleven)."""
    eleven = FakeEleven(script, on_commit)

    async def connect(url, additional_headers=None, **kwargs):
        eleven.url, eleven.headers = url, additional_headers
        return eleven

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = el_stt.App()
        await app.setup()
        yields = []
        try:
            async for out in app.realtime(el_stt.RealtimeInput(**input_fields), caller):
                yields.append(out)
        except Exception as err:  # noqa: BLE001 - the error is the result under test
            return yields, err
        return yields, None

    yields, err = asyncio.run(go())
    return yields, err, eleven


# ------------------------------------------------------------ the transcript


def test_a_partial_is_replaced_and_a_commit_is_kept():
    script = [
        READY, partial("what's"), partial("what's the wea"), committed("What's the weather?"),
        timed("What's the weather?", 0.4, 1.9), partial("and to"), "hang",
    ]
    caller = FakeCaller(close_after=0.4)
    yields, err, _ = realtime(script, caller)

    assert err is None
    assert caller.texts() == ["", "what's", "what's the wea", "What's the weather?", "What's the weather? and to"]
    result = yields[-1]
    assert [s.text for s in result.segments] == ["What's the weather?", "and to"], "the uncommitted tail is kept"
    assert (result.segments[0].start, result.segments[0].end) == (0.4, 1.9), "timed by its words, not the spacing"
    assert result.language_code == "en"


def test_a_pause_committed_with_nothing_said_adds_no_segment():
    yields, _, _ = realtime([READY, committed(""), committed("One."), committed(""), "hang"], FakeCaller(close_after=0.3))

    assert [(y.text, y.partial) for y in yields] == [("One.", True), ("One.", False)]
    assert len(yields[-1].segments) == 1


# ------------------------------------------------------------ ending a session


def test_the_caller_closing_commits_the_tail_and_bills_the_audio_sent():
    caller = FakeCaller(frames=[FRAME] * 7, close_after=0.1)
    on_commit = [committed("Goodbye."), timed("Goodbye.", 0.2, 0.9)]
    yields, err, eleven = realtime([READY, partial("good"), "hang"], caller, on_commit=on_commit)

    assert err is None
    assert eleven.sent[-1] == {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True, "sample_rate": 16000}
    assert b"".join(eleven.audio()) == FRAME * 7, "the frames left over from the last chunk are sent before the commit"
    result = yields[-1]
    assert (result.text, result.partial, result.end_reason) == ("Goodbye.", False, "the caller closed the session")
    assert (result.segments[0].start, result.segments[0].end) == (0.2, 0.9), "the timings of the last commit are waited for"
    meta = result.output_meta.inputs[0]
    assert meta.seconds == 0.14
    assert meta.extra == {"model": "scribe_v2_realtime", "keyterms": False, "caller_seconds": 0.14}


def test_elevenlabs_not_answering_the_commit_still_ends_with_a_billed_result(monkeypatch):
    monkeypatch.setattr(el_stt, "FLUSH_SECONDS", 0.2)
    caller = FakeCaller(frames=[FRAME] * 50, close_after=0.1)
    yields, err, _ = realtime([READY, committed("Hello."), partial("and"), "hang"], caller, on_commit=[])

    assert err is None
    assert yields[-1].text == "Hello. and"
    assert yields[-1].output_meta.inputs[0].seconds == 1.0


def test_elevenlabs_closing_a_started_session_is_a_normal_billed_end():
    limit = {"message_type": "session_time_limit_exceeded", "error": "Maximum session time has been reached."}
    caller = FakeCaller(frames=[FRAME] * 10)
    yields, err, _ = realtime([READY, committed("Hello."), limit, "close"], caller)

    assert err is None
    result = yields[-1]
    assert (result.partial, result.end_reason) == (False, "ElevenLabs: Maximum session time has been reached.")
    assert result.text == "Hello."
    assert result.output_meta.inputs[0].extra["model"] == "scribe_v2_realtime", "it ends with its usage"
    assert caller.errors() == ["ElevenLabs: Maximum session time has been reached."]


def test_elevenlabs_refusing_before_starting_fails_the_task():
    refused = {"message_type": "auth_error", "error": "Invalid API key"}
    yields, err, _ = realtime([refused, "close"], FakeCaller())

    assert yields == []
    assert isinstance(err, RuntimeError) and "Invalid API key" in str(err)


def test_a_throttled_commit_reaches_the_caller_and_the_session_goes_on():
    throttled = {"message_type": "commit_throttled", "error": "Commit throttled"}
    caller = FakeCaller(close_after=0.3)
    yields, err, _ = realtime([READY, throttled, committed("Still here."), "hang"], caller)

    assert err is None
    assert caller.errors() == ["ElevenLabs: Commit throttled"]
    assert (yields[-1].text, yields[-1].end_reason) == ("Still here.", "the caller closed the session")


def test_nobody_speaking_ends_the_session_with_a_result():
    caller = FakeCaller(frames=[FRAME] * 5)
    yields, err, _ = realtime([READY, "hang"], caller, idle_minutes=0.002)

    assert err is None
    assert yields[-1].end_reason == "ended after 0.002 minutes with nobody speaking"
    assert caller.errors() == ["ended after 0.002 minutes with nobody speaking"]
    assert yields[-1].output_meta.inputs[0].seconds > 0


# ------------------------------------------------------------ what ElevenLabs is sent


def test_audio_is_sent_in_tenth_of_a_second_chunks():
    frames = [bytes([i, 0]) * 320 for i in range(1, 12)]        # 11 frames of 20 ms
    _, _, eleven = realtime([READY, "hang"], FakeCaller(frames=frames, close_after=0.05))

    chunks = [a for a in eleven.audio() if a]
    assert [len(c) for c in chunks] == [3200, 3200, 640]
    assert b"".join(chunks) == b"".join(frames)
    assert all(m["sample_rate"] == 16000 and m["message_type"] == "input_audio_chunk" for m in eleven.sent)


def test_a_pause_is_sent_as_silence(monkeypatch):
    monkeypatch.setattr(el_stt, "GAP_SECONDS", 0.1)
    yields, _, eleven = realtime([READY, "hang"], FakeCaller(frames=[FRAME], close_after=0.5))

    silence = [a for a in eleven.audio() if a == el_stt.SILENCE]
    assert 2 <= len(silence) <= 5, "100 ms of silence per 100 ms of pause, after the gap"
    assert eleven.sent[-1]["commit"] is True and eleven.sent[-2]["audio_base_64"] != "", "none after the caller closed"
    assert yields[-1].seconds == round((len(FRAME) + len(silence) * 3200) / 32000, 3)


def test_the_options_go_in_the_url_and_the_key_in_a_header():
    _, _, eleven = realtime(
        [READY, "hang"], FakeCaller(close_after=0.05), language_code="en", keyterms=["Scribe", "xi"], silence_ms=800,
    )

    assert eleven.url == (
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime&audio_format=pcm_16000"
        "&commit_strategy=vad&vad_silence_threshold_secs=0.8&include_timestamps=true&language_code=en"
        "&keyterms=Scribe&keyterms=xi"
    )
    assert eleven.headers == {"xi-api-key": "test"}


def test_the_language_is_detected_when_none_is_given_and_keyterms_are_reported_for_billing():
    yields, _, eleven = realtime([READY, "hang"], FakeCaller(close_after=0.05), keyterms=["Scribe"])

    assert "include_language_detection=true" in eleven.url and "language_code" not in eleven.url
    assert yields[-1].output_meta.inputs[0].extra["keyterms"] is True
