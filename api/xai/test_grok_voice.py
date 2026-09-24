"""No-network tests for grok-voice: how a session ends and is billed, the
transcript, and what the app asks Grok for.

Grok is a scripted fake socket; the caller is a fake platform socket.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("XAI_API_KEY", "test")

_spec = importlib.util.spec_from_file_location("grok_voice", Path(__file__).with_name("grok-voice") / "inference.py")
grok_voice = importlib.util.module_from_spec(_spec)
sys.modules["grok_voice"] = grok_voice
_spec.loader.exec_module(grok_voice)

READY = {"type": "session.updated"}
TIMEOUT = {"type": "error", "error": {"type": "timeout", "message": "Conversation timed out after 900.0 seconds due to inactivity"}}


class FakeGrok:
    """Grok's realtime socket: plays a script of events (dicts, raw bytes,
    "close" to end the socket, "hang" to go quiet) and records what it was sent."""

    def __init__(self, script):
        self.queue = asyncio.Queue()
        for item in script:
            self.queue.put_nowait(item)
        self.sent = []
        self.url = None

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        await asyncio.sleep(0.01)
        if item == "close":
            raise StopAsyncIteration
        if item == "hang":
            await asyncio.sleep(3600)
        return item if isinstance(item, bytes) else json.dumps(item)

    async def close(self):
        pass

    def events_sent(self):
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

    def patches(self):
        return [o for o in self.out if isinstance(o, dict)]


def talk(script, caller, **input_fields):
    """Runs one talk session; returns (yields, error, grok)."""
    grok = FakeGrok(script)

    async def connect(url, **kwargs):
        grok.url = url
        return grok

    sys.modules["websockets"] = types.SimpleNamespace(connect=connect)

    async def go():
        app = grok_voice.App()
        await app.setup(None)
        yields = []
        try:
            async for out in app.talk(grok_voice.TalkInput(**input_fields), caller):
                yields.append(out)
        except Exception as err:  # noqa: BLE001 - the error is the result under test
            return yields, err
        return yields, None

    yields, err = asyncio.run(go())
    return yields, err, grok


# ------------------------------------------------------------ ending a session


def test_grok_ending_an_accepted_session_is_a_normal_billed_end():
    yields, err, _ = talk([READY, TIMEOUT, "close"], FakeCaller())

    assert err is None
    result = yields[-1]
    assert result.partial is False
    assert result.end_reason.startswith("Grok: Conversation timed out")
    assert result.output_meta.inputs[0].seconds > 0, "the session's time is billed"


def test_grok_refusing_before_accepting_fails_the_task():
    yields, err, _ = talk([TIMEOUT, "close"], FakeCaller())

    assert yields == []
    assert isinstance(err, RuntimeError) and "timed out" in str(err)


def test_a_non_fatal_grok_error_reaches_the_caller_and_the_session_goes_on():
    bad = {"type": "error", "error": {"message": "unknown voice"}}
    caller = FakeCaller(close_after=0.3)
    yields, err, _ = talk([READY, bad, "hang"], caller)

    assert err is None
    assert yields[-1].end_reason == "the caller closed the session"
    assert {"error": {"field": None, "message": "Grok: unknown voice"}} in caller.patches()


def test_the_caller_is_told_why_grok_ended_the_session():
    caller = FakeCaller()
    talk([READY, TIMEOUT, "close"], caller)

    errors = [p["error"]["message"] for p in caller.patches() if "error" in p]
    assert errors == ["Grok: Conversation timed out after 900.0 seconds due to inactivity"]


# ------------------------------------------------------------------ transcript


def test_the_conversation_is_the_result_typed_and_spoken_turns_in_order():
    script = [
        READY,
        {"type": "response.created"},
        {"type": "response.output_audio_transcript.delta", "delta": "Hi "},
        b"\x00\x01" * 480,
        {"type": "response.output_audio_transcript.delta", "delta": "there."},
        {"type": "response.done", "response": {}},
        {"type": "conversation.item.input_audio_transcription.updated", "item_id": "u1", "content": "what's"},
        {"type": "conversation.item.input_audio_transcription.updated", "item_id": "u1", "content": "what's up"},
        {"type": "response.created"},
        {"type": "response.output_audio_transcript.delta", "delta": "Not much."},
        {"type": "response.done", "response": {}},
        "hang",
    ]
    caller = FakeCaller([json.dumps({"events": {"type": "text", "text": "hello"}})], close_after=0.6)
    yields, err, _ = talk(script, caller)

    assert err is None
    assert [(m.role, m.text) for m in yields[-1].messages] == [
        ("user", "hello"),
        ("assistant", "Hi there."),
        ("user", "what's up"),
        ("assistant", "Not much."),
    ]
    assert yields[-1].output_meta.outputs[0].seconds > 0, "the assistant's audio is reported"
    assert any(isinstance(o, bytes) for o in caller.out), "Grok's audio reaches the caller untouched"


def test_repeated_user_transcripts_are_sent_once():
    same = {"type": "conversation.item.input_audio_transcription.updated", "item_id": "u1", "content": ""}
    caller = FakeCaller(close_after=0.3)
    talk([READY, same, same, same, "hang"], caller)

    assert [p for p in caller.patches() if "user_text" in p] == [{"user_text": ""}], "only the first frame, no repeats"


# ------------------------------------------------------------- what Grok is asked


def test_session_asks_for_binary_pcm_both_ways_and_the_input_settings():
    _, _, grok = talk([READY, "hang"], FakeCaller(close_after=0.1), voice="ara", reasoning="none", language="ja", silence_ms=500, web_search=True)

    session = grok.events_sent()[0]["session"]
    assert session["voice"] == "ara"
    assert session["reasoning"] == {"effort": "none"}
    assert session["turn_detection"] == {"type": "server_vad", "silence_duration_ms": 500}
    assert session["tools"] == [{"type": "web_search"}]
    audio = session["audio"]
    assert audio["input"]["format"] == {"type": "audio/pcm", "rate": 24000} and audio["input"]["transport"] == "binary"
    assert audio["output"]["transport"] == "binary"
    assert audio["input"]["transcription"] == {"model": "grok-transcribe", "language_hint": "ja"}
    assert grok.url.endswith("?model=grok-voice-latest")


def test_a_custom_voice_replaces_the_built_in_one():
    _, _, grok = talk([READY, "hang"], FakeCaller(close_after=0.1), voice="eve", custom_voice="my-clone")

    assert grok.events_sent()[0]["session"]["voice"] == "my-clone"


def test_changing_a_field_mid_stream_updates_the_grok_session():
    caller = FakeCaller([json.dumps({"voice": "rex"})], close_after=0.2)
    _, _, grok = talk([READY, "hang"], caller)

    updates = [e for e in grok.events_sent() if e["type"] == "session.update"]
    assert [u["session"]["voice"] for u in updates] == ["eve", "rex"]


def test_typed_items_become_grok_items():
    caller = FakeCaller([
        json.dumps({"events": {"type": "text", "text": "hi"}}),
        json.dumps({"events": {"type": "say", "text": "This call is recorded.", "interruptible": False}}),
    ], close_after=0.2)
    _, _, grok = talk([READY, "hang"], caller)

    sent = [e for e in grok.events_sent() if e["type"] != "session.update"]
    assert sent[0]["item"] == {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    assert sent[1] == {"type": "response.create"}, "a typed user turn asks for an answer"
    assert sent[2]["item"]["type"] == "force_message" and sent[2]["item"]["interruptible"] is False
    assert len(sent) == 3, "a verbatim line is its own turn: no response.create"


def test_billing_counts_each_typed_message_as_a_text_input():
    caller = FakeCaller([json.dumps({"events": {"type": "text", "text": t}}) for t in ("a", "b")], close_after=0.2)
    yields, _, _ = talk([READY, "hang"], caller)

    inputs = yields[-1].output_meta.inputs
    assert [i.type for i in inputs] == ["audio", "text", "text"]
