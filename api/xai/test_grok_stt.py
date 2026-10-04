"""No-network tests for grok-stt: the transcript xAI's results add up to, how a
live session ends and is billed, and what the app asks xAI for.

xAI is a scripted fake socket; the caller is a fake platform socket.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("XAI_API_KEY", "test")

_spec = importlib.util.spec_from_file_location("grok_stt", Path(__file__).with_name("grok-stt") / "inference.py")
grok_stt = importlib.util.module_from_spec(_spec)
sys.modules["grok_stt"] = grok_stt
_spec.loader.exec_module(grok_stt)

READY = {"type": "transcript.created", "id": "t"}
FRAME = b"\x01\x00" * 320        # 20 ms of 16 kHz PCM


def partial(text, start=0.0, duration=1.0, final=False, speech_final=False, words=None):
    return {
        "type": "transcript.partial", "text": text, "words": words or [], "is_final": final,
        "speech_final": speech_final, "start": start, "duration": duration,
    }


class FakeXai:
    """xAI's streaming socket: plays a script of events ("close" ends the
    socket, "hang" goes quiet), answers ``audio.done`` with `on_done`, and
    records what it was sent."""

    def __init__(self, script, on_done=()):
        self.queue = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)
        self.on_done = list(on_done)
        self.sent = []
        self.url = None

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, str) and json.loads(data).get("type") == "audio.done":
            for item in self.on_done:
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
                continue                    # nothing more until something else is queued
            return json.dumps(item)

    async def close(self):
        pass

    def audio(self):
        return [m for m in self.sent if isinstance(m, bytes)]

    def messages(self):
        return [json.loads(m) for m in self.sent if isinstance(m, str)]


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


DONE = [{"type": "transcript.done", "text": "", "words": [], "duration": 0}, "close"]


def realtime(script, caller, on_done=DONE, **input_fields):
    """Runs one realtime session; returns (yields, error, xai)."""
    xai = FakeXai(script, on_done)

    async def connect(url, **kwargs):
        xai.url = url
        return xai

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = grok_stt.App()
        await app.setup(None)
        yields = []
        try:
            async for out in app.realtime(grok_stt.RealtimeInput(**input_fields), caller):
                yields.append(out)
        except Exception as err:  # noqa: BLE001 - the error is the result under test
            return yields, err
        return yields, None

    yields, err = asyncio.run(go())
    return yields, err, xai


# ------------------------------------------------------------ the transcript


def test_an_interim_result_is_replaced_and_a_final_one_is_kept():
    script = [
        READY,
        partial("what's"),
        partial("what's the wea"),
        partial("What's the weather?", final=True, speech_final=True, duration=1.4),
        partial("and to"),
        "hang",
    ]
    caller = FakeCaller(close_after=0.4)
    yields, err, _ = realtime(script, caller)

    assert err is None
    assert caller.texts() == ["", "what's", "what's the wea", "What's the weather?", "What's the weather? and to"]
    result = yields[-1]
    assert [u.text for u in result.utterances] == ["What's the weather?", "and to"], "the unsettled tail is kept"
    assert result.utterances[0].end == 1.4


def test_the_stitched_utterance_replaces_the_stretches_it_covers():
    script = [
        READY,
        partial("The balance is", start=0.0, duration=3.0, final=True),
        partial("one hundred dollars", start=3.0, duration=2.0),
        partial("The balance is $100.", start=0.0, duration=5.2, final=True, speech_final=True),
        "hang",
    ]
    yields, _, _ = realtime(script, FakeCaller(close_after=0.3))

    result = yields[-1]
    assert result.text == "The balance is $100."
    assert [(u.start, u.end) for u in result.utterances] == [(0.0, 5.2)]


def test_a_final_that_only_covers_the_last_stretch_is_added_to_the_ones_before():
    script = [
        READY,
        partial("The balance is", start=0.0, duration=3.0, final=True),
        partial("one hundred dollars.", start=3.0, duration=2.0, final=True, speech_final=True),
        "hang",
    ]
    yields, _, _ = realtime(script, FakeCaller(close_after=0.3))

    result = yields[-1]
    assert result.text == "The balance is one hundred dollars."
    assert [(u.start, u.end) for u in result.utterances] == [(0.0, 5.0)]


def test_an_utterance_carries_the_speaker_who_said_most_of_it():
    words = [{"text": "hi", "speaker": 1}, {"text": "there", "speaker": 1}, {"text": "ok", "speaker": 0}]
    script = [READY, partial("hi there ok", final=True, speech_final=True, words=words), "hang"]
    yields, _, _ = realtime(script, FakeCaller(close_after=0.3), diarize=True)

    assert yields[-1].utterances[0].speaker == 1


def test_every_settled_utterance_is_a_snapshot_and_the_result_is_last():
    script = [
        READY,
        partial("One.", final=True, speech_final=True),
        partial("Two.", start=2.0, final=True, speech_final=True),
        "hang",
    ]
    yields, _, _ = realtime(script, FakeCaller(close_after=0.3))

    assert [(y.text, y.partial) for y in yields] == [("One.", True), ("One. Two.", True), ("One. Two.", False)]


# ------------------------------------------------------------ ending a session


def test_the_caller_closing_flushes_the_tail_and_bills_what_xai_counted():
    on_done = [
        partial("Goodbye.", start=1.0, duration=0.8, final=True, speech_final=True),
        {"type": "transcript.done", "text": "", "words": [], "duration": 6.43},
        "close",
    ]
    yields, err, xai = realtime([READY, "hang"], FakeCaller(frames=[FRAME] * 5, close_after=0.1), on_done=on_done)

    assert err is None
    assert xai.messages() == [{"type": "audio.done"}]
    result = yields[-1]
    assert (result.text, result.partial, result.end_reason) == ("Goodbye.", False, "the caller closed the session")
    meta = result.output_meta.inputs[0]
    assert meta.seconds == 6.43, "xAI's own count of the audio is what is billed"
    assert meta.extra["caller_seconds"] == 0.1


def test_xai_not_answering_the_flush_still_ends_with_a_billed_result(monkeypatch):
    monkeypatch.setattr(grok_stt, "FLUSH_SECONDS", 0.2)
    script = [READY, partial("Hello.", final=True, speech_final=True), "hang"]
    yields, err, _ = realtime(script, FakeCaller(frames=[FRAME] * 50, close_after=0.1), on_done=[])

    assert err is None
    result = yields[-1]
    assert result.text == "Hello."
    assert result.output_meta.inputs[0].seconds == 1.0, "without xAI's count, the audio sent is billed"


def test_xai_closing_an_accepted_stream_is_a_normal_billed_end():
    script = [READY, partial("Hello.", final=True, speech_final=True), "close"]
    yields, err, _ = realtime(script, FakeCaller(frames=[FRAME] * 10))

    assert err is None
    result = yields[-1]
    assert (result.partial, result.end_reason) == (False, "xAI closed the stream")
    assert result.text == "Hello."
    assert result.output_meta.inputs[0].seconds > 0


def test_xai_refusing_before_accepting_fails_the_task():
    refused = {"type": "error", "message": "unsupported sample rate"}
    yields, err, _ = realtime([refused, "close"], FakeCaller())

    assert yields == []
    assert isinstance(err, RuntimeError) and "unsupported sample rate" in str(err)


def test_an_xai_error_reaches_the_caller():
    caller = FakeCaller(close_after=0.3)
    yields, err, _ = realtime([READY, {"type": "error", "message": "Invalid message"}, "hang"], caller)

    assert err is None
    assert caller.errors() == ["xAI: Invalid message"]
    assert yields[-1].end_reason == "the caller closed the session"


def test_nobody_speaking_ends_the_session_with_a_result():
    caller = FakeCaller(frames=[FRAME] * 5)
    yields, err, _ = realtime([READY, "hang"], caller, idle_minutes=0.002)

    assert err is None
    assert yields[-1].end_reason == "ended after 0.002 minutes with nobody speaking"
    assert caller.errors() == ["ended after 0.002 minutes with nobody speaking"]
    assert yields[-1].output_meta.inputs[0].seconds > 0


def test_idle_minutes_zero_never_ends_the_session():
    yields, _, _ = realtime([READY, "hang"], FakeCaller(close_after=0.5), idle_minutes=0)

    assert yields[-1].end_reason == "the caller closed the session"


# ------------------------------------------------------------ what xAI is sent


def test_audio_passes_through_untouched_once_xai_is_ready():
    frames = [bytes([i, 0]) * 320 for i in range(1, 4)]
    _, _, xai = realtime([READY, "hang"], FakeCaller(frames=frames, close_after=0.05))

    assert xai.audio() == frames


def test_a_pause_is_sent_as_silence(monkeypatch):
    monkeypatch.setattr(grok_stt, "GAP_SECONDS", 0.1)
    yields, _, xai = realtime([READY, "hang"], FakeCaller(frames=[FRAME], close_after=0.5))

    silence = [m for m in xai.audio() if m == grok_stt.SILENCE]
    assert 2 <= len(silence) <= 5, "100 ms of silence per 100 ms of pause, after the gap"
    assert xai.sent[-1] == json.dumps({"type": "audio.done"}), "none after the caller closed"
    assert yields[-1].seconds == round((len(FRAME) + len(silence) * len(grok_stt.SILENCE)) / 32000, 3)


def test_the_options_go_in_the_stream_url():
    _, _, xai = realtime(
        [READY, "hang"], FakeCaller(close_after=0.05),
        language="en", diarize=True, keyterms=["Grok", "xAI"], silence_ms=800,
    )

    assert xai.url == (
        "wss://api.x.ai/v1/stt?model=grok-voice-transcribe-2.0&sample_rate=16000&encoding=pcm"
        "&interim_results=true&endpointing=800&diarize=true&filler_words=false&language=en&keyterm=Grok&keyterm=xAI"
    )


# ------------------------------------------------------------ a recording


class FakeHttp:
    """httpx.AsyncClient, answering one POST and recording it."""

    def __init__(self, status, body):
        self.status, self.body, self.request = status, body, None

    def __call__(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, data=None, files=None):
        self.request = {"url": url, "headers": headers, "data": data, "file": files["file"][0]}
        return types.SimpleNamespace(status_code=self.status, text=json.dumps(self.body), json=lambda: self.body)


def run(tmp_path, monkeypatch, status=200, body=None, **input_fields):
    http = FakeHttp(status, body or {})
    monkeypatch.setattr(grok_stt.httpx, "AsyncClient", http)
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"RIFF")

    async def go():
        app = grok_stt.App()
        await app.setup(None)
        return await app.run(grok_stt.AppInput(audio=str(audio), **input_fields))

    try:
        return asyncio.run(go()), None, http
    except Exception as err:  # noqa: BLE001 - the error is the result under test
        return None, err, http


def test_a_recording_comes_back_with_its_words_and_is_billed_by_its_length(tmp_path, monkeypatch):
    body = {
        "text": "The balance.", "language": "en", "duration": 8.4,
        "words": [{"text": "The", "start": 0, "end": 0.24, "speaker": 0}, {"text": "balance.", "start": 0.24, "end": 0.64, "confidence": 0.67}],
    }
    out, err, http = run(tmp_path, monkeypatch, body=body, language="en", diarize=True, keyterms=["Grok", "xAI"])

    assert err is None
    assert (out.text, out.language, out.duration) == ("The balance.", "en", 8.4)
    assert [(w.text, w.speaker, w.confidence) for w in out.words] == [("The", 0, None), ("balance.", None, 0.67)]
    assert out.output_meta.inputs[0].seconds == 8.4
    assert http.request["data"] == {
        "model": "grok-voice-transcribe-2.0", "diarize": "true", "filler_words": "false",
        "language": "en", "format": "true", "keyterm": ["Grok", "xAI"],
    }
    assert http.request["file"] == "talk.wav"


def test_formatting_is_only_asked_for_with_a_language(tmp_path, monkeypatch):
    _, _, http = run(tmp_path, monkeypatch, body={"text": "", "duration": 1})

    assert "format" not in http.request["data"] and "language" not in http.request["data"]


def test_an_xai_refusal_fails_the_run_with_what_xai_said(tmp_path, monkeypatch):
    out, err, _ = run(tmp_path, monkeypatch, status=400, body={"error": "unsupported format"})

    assert out is None
    assert isinstance(err, RuntimeError) and "400" in str(err) and "unsupported format" in str(err)
