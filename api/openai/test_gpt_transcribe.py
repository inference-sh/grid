"""No-network tests for gpt-transcribe: the transcript OpenAI's events add up
to, where the app ends a turn, how a live session ends and is billed, and what
the app asks OpenAI for.

OpenAI is a scripted fake socket; the caller is a fake platform socket.
"""

import asyncio
import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OPENAI_KEY", "test")

_spec = importlib.util.spec_from_file_location("gpt_transcribe", Path(__file__).with_name("gpt-transcribe") / "inference.py")
gpt_transcribe = importlib.util.module_from_spec(_spec)
sys.modules["gpt_transcribe"] = gpt_transcribe
_spec.loader.exec_module(gpt_transcribe)

READY = {"type": "session.updated"}
LOUD = b"\x00\x20" * 480         # 20 ms of 24 kHz PCM, loud enough to be speech
QUIET = b"\x00\x00" * 480        # 20 ms of silence
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"


def delta(item, text):
    return {"type": DELTA, "item_id": item, "content_index": 0, "delta": text}


def completed(item, text):
    return {"type": COMPLETED, "item_id": item, "content_index": 0, "transcript": text}


def committed(item):
    return {"type": "input_audio_buffer.committed", "item_id": item}


class FakeOpenAI:
    """OpenAI's realtime socket: plays a script of events ("close" ends the
    socket, "hang" goes quiet), answers each commit with the next entry of
    `on_commit`, and records what it was sent."""

    def __init__(self, script, on_commit=()):
        self.queue = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)
        self.on_commit = [list(answer) for answer in on_commit]
        self.sent = []
        self.url = None
        self.headers = None

    async def send(self, data):
        message = json.loads(data)
        self.sent.append(message)
        if message["type"] == "input_audio_buffer.commit" and self.on_commit:
            for item in self.on_commit.pop(0):
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

    def kinds(self):
        return [m["type"] for m in self.sent]

    def audio(self):
        return b"".join(base64.b64decode(m["audio"]) for m in self.sent if m["type"] == "input_audio_buffer.append")


class FakeCaller:
    """The platform socket the kernel hands the app: frames from the caller
    (bytes, a patch as a dict, or a number of seconds to send nothing), then
    the caller closes after `close_after` seconds of nothing left."""

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
            if isinstance(frame, (int, float)):
                await asyncio.sleep(frame)
                continue
            await asyncio.sleep(0.002)
            yield json.dumps(frame) if isinstance(frame, dict) else frame
        await asyncio.sleep(self.close_after)

    async def send(self, data):
        self.out.append(data)

    async def close(self):
        self.closed = True

    def texts(self):
        return [o["text"] for o in self.out if isinstance(o, dict) and "text" in o]

    def errors(self):
        return [o["$error"]["message"] for o in self.out if isinstance(o, dict) and "$error" in o]


def realtime(script, caller, on_commit=(), **input_fields):
    """Runs one realtime session; returns (yields, error, openai)."""
    openai = FakeOpenAI(script, on_commit)

    async def connect(url, additional_headers=None, **kwargs):
        openai.url, openai.headers = url, additional_headers
        return openai

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = gpt_transcribe.App()
        await app.setup(None)
        yields = []
        try:
            async for out in app.realtime(gpt_transcribe.RealtimeInput(**input_fields), caller):
                yields.append(out)
        except Exception as err:  # noqa: BLE001 - the error is the result under test
            return yields, err
        return yields, None

    yields, err = asyncio.run(go())
    return yields, err, openai


# ------------------------------------------------------------ the transcript


def test_words_add_up_as_they_are_heard_and_the_final_text_replaces_them():
    script = [
        READY, delta("a", "what's"), delta("a", " the wea"), committed("a"), completed("a", "What's the weather?"),
        delta("b", "and to"), "hang",
    ]
    caller = FakeCaller(close_after=0.4)
    yields, err, _ = realtime(script, caller)

    assert err is None
    assert caller.texts() == ["", "what's", "what's the wea", "What's the weather?", "What's the weather? and to"]
    result = yields[-1]
    assert [t.text for t in result.turns] == ["What's the weather?", "and to"], "what was heard of the open turn is kept"


def test_final_texts_arriving_out_of_order_keep_the_order_of_speech():
    script = [
        READY, delta("a", "one"), committed("a"), delta("b", "two"), committed("b"),
        completed("b", "Two."), completed("a", "One."), "hang",
    ]
    yields, _, _ = realtime(script, FakeCaller(close_after=0.3))

    assert yields[-1].text == "One. Two."
    assert [t.text for t in yields[-1].turns] == ["One.", "Two."]


def test_a_turn_openai_could_not_transcribe_keeps_what_was_heard_of_it():
    failed = {"type": "conversation.item.input_audio_transcription.failed", "item_id": "a", "error": {"code": "audio_unintelligible"}}
    yields, err, _ = realtime([READY, delta("a", "hello"), committed("a"), failed, "hang"], FakeCaller(close_after=0.3))

    assert err is None
    assert yields[-1].text == "hello"


# ------------------------------------------------------------ ending a turn


def test_a_pause_after_speech_commits_the_turn():
    # Half a second of speech, a pause, then more speech: two turns.
    frames = [LOUD] * 25 + [QUIET] * 10 + [0.45] + [LOUD] * 25
    on_commit = [[committed("a"), completed("a", "Hello.")], [committed("b"), completed("b", "Again.")]]
    yields, _, openai = realtime([READY, "hang"], FakeCaller(frames=frames, close_after=0.05), on_commit=on_commit, silence_ms=300)

    assert openai.kinds().count("input_audio_buffer.commit") == 2, "one at the pause, one when the caller closed"
    first, second = yields[-1].turns
    assert (first.text, first.start, second.text) == ("Hello.", 0.0, "Again.")
    assert 0.5 <= first.end <= 0.7, "the turn spans the audio sent before the pause"
    assert second.start == first.end and second.end == 1.2


def test_silence_alone_is_not_a_turn():
    _, _, openai = realtime([READY, "hang"], FakeCaller(frames=[QUIET] * 40 + [0.5], close_after=0.1), silence_ms=200)

    assert "input_audio_buffer.commit" not in openai.kinds()


def test_a_turn_that_never_pauses_is_committed_at_the_limit(monkeypatch):
    monkeypatch.setattr(gpt_transcribe, "MAX_TURN_SECONDS", 0.4)
    _, _, openai = realtime([READY, "hang"], FakeCaller(frames=[LOUD] * 60, close_after=0.1))

    assert openai.kinds().count("input_audio_buffer.commit") >= 1


# ------------------------------------------------------------ ending a session


def test_the_caller_closing_commits_the_open_turn_and_bills_the_audio_sent():
    on_commit = [[committed("a"), completed("a", "Goodbye.")]]
    caller = FakeCaller(frames=[LOUD] * 17, close_after=0.05)
    yields, err, openai = realtime([READY, delta("a", "good"), "hang"], caller, on_commit=on_commit)

    assert err is None
    assert openai.kinds()[-1] == "input_audio_buffer.commit"
    assert openai.audio() == LOUD * 17, "the frames left over from the last chunk are sent before the commit"
    result = yields[-1]
    assert (result.text, result.partial, result.end_reason) == ("Goodbye.", False, "the caller closed the session")
    meta = result.output_meta.inputs[0]
    assert (meta.seconds, meta.extra) == (0.34, {"model": "gpt-live-transcribe"})


def test_openai_not_answering_the_last_commit_still_ends_with_a_billed_result(monkeypatch):
    monkeypatch.setattr(gpt_transcribe, "FLUSH_SECONDS", 0.2)
    caller = FakeCaller(frames=[LOUD] * 50, close_after=0.05)
    yields, err, _ = realtime([READY, delta("a", "hello and"), "hang"], caller)

    assert err is None
    assert yields[-1].text == "hello and"
    assert yields[-1].output_meta.inputs[0].seconds == 1.0


def test_openai_closing_an_accepted_session_is_a_normal_billed_end():
    yields, err, _ = realtime([READY, delta("a", "Hello."), "close"], FakeCaller(frames=[LOUD] * 10))

    assert err is None
    result = yields[-1]
    assert (result.partial, result.end_reason, result.text) == (False, "OpenAI closed the session", "Hello.")
    assert result.output_meta.inputs[0].extra == {"model": "gpt-live-transcribe"}


def test_a_session_openai_rejects_fails_the_task():
    rejected = {"type": "error", "error": {"type": "invalid_request_error", "message": "Unsupported language code 'xx'."}}
    yields, err, _ = realtime([{"type": "session.created"}, rejected, "hang"], FakeCaller(), languages=["xx"])

    assert yields == []
    assert isinstance(err, RuntimeError) and "Unsupported language code" in str(err)


def test_a_recoverable_openai_error_reaches_the_caller_and_the_session_goes_on():
    bad = {"type": "error", "error": {"message": "buffer too small"}}
    caller = FakeCaller(close_after=0.3)
    yields, err, _ = realtime([READY, bad, "hang"], caller)

    assert err is None
    assert caller.errors() == ["OpenAI: buffer too small"]
    assert yields[-1].end_reason == "the caller closed the session"


def test_nobody_speaking_ends_the_session_with_a_result():
    caller = FakeCaller(frames=[QUIET] * 5)
    yields, err, _ = realtime([READY, "hang"], caller, idle_minutes=0.002)

    assert err is None
    assert yields[-1].end_reason == "ended after 0.002 minutes with nobody speaking"
    assert caller.errors() == ["ended after 0.002 minutes with nobody speaking"]


# ------------------------------------------------------------ what OpenAI is sent


def test_the_session_is_a_transcription_session_without_turn_detection():
    _, _, openai = realtime(
        [READY, "hang"], FakeCaller(close_after=0.05),
        prompt="A support call.", languages=["en", "fr"], keyterms=["AC-42"], delay="low",
    )

    assert openai.url == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert openai.headers == {"Authorization": "Bearer test"}
    assert openai.sent[0] == {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {
                    "model": "gpt-live-transcribe", "prompt": "A support call.",
                    "languages": ["en", "fr"], "keywords": ["AC-42"], "delay": "low",
                },
                "turn_detection": None,
            }},
        },
    }


def test_changing_the_keyterms_mid_stream_updates_the_session():
    caller = FakeCaller(frames=[{"keyterms": ["AC-42"]}], close_after=0.1)
    _, _, openai = realtime([READY, "hang"], caller)

    updates = [m for m in openai.sent if m["type"] == "session.update"]
    assert len(updates) == 2
    assert updates[1]["session"]["audio"]["input"]["transcription"]["keywords"] == ["AC-42"]


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


def run(tmp_path, monkeypatch, status=200, body=None, measured=0.0, **input_fields):
    http = FakeHttp(status, body or {})
    monkeypatch.setattr(gpt_transcribe.httpx, "AsyncClient", http)
    monkeypatch.setattr(gpt_transcribe, "_audio_seconds", lambda path: measured)
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"RIFF")

    async def go():
        app = gpt_transcribe.App()
        await app.setup(None)
        return await app.run(gpt_transcribe.AppInput(audio=str(audio), **input_fields))

    try:
        return asyncio.run(go()), None, http
    except Exception as err:  # noqa: BLE001 - the error is the result under test
        return None, err, http


def test_a_recording_is_billed_by_the_seconds_openai_reports(tmp_path, monkeypatch):
    body = {"text": "Bonjour.", "languages": [{"code": "fr"}], "usage": {"type": "duration", "seconds": 27}}
    out, err, http = run(tmp_path, monkeypatch, body=body, measured=99.0, languages=["fr", "en"], keyterms=["AC-42"], prompt="A call.")

    assert err is None
    assert (out.text, out.languages, out.duration) == ("Bonjour.", ["fr"], 27.0)
    assert out.output_meta.inputs[0].seconds == 27.0
    assert http.request["data"] == {
        "model": "gpt-transcribe", "prompt": "A call.", "languages[]": ["fr", "en"], "keywords[]": ["AC-42"],
    }
    assert http.request["headers"] == {"Authorization": "Bearer test"}


def test_a_recording_whose_usage_is_in_tokens_is_measured_here(tmp_path, monkeypatch):
    body = {"text": "Hello.", "usage": {"type": "tokens", "input_tokens": 14, "output_tokens": 45, "total_tokens": 59}}
    out, _, _ = run(tmp_path, monkeypatch, body=body, measured=8.5)

    assert out.output_meta.inputs[0].seconds == 8.5


def test_an_openai_refusal_fails_the_run_with_what_openai_said(tmp_path, monkeypatch):
    out, err, _ = run(tmp_path, monkeypatch, status=400, body={"error": {"message": "file too large"}})

    assert out is None
    assert isinstance(err, RuntimeError) and "400" in str(err) and "file too large" in str(err)
