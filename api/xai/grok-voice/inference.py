"""Grok Voice: a live conversation with xAI's speech-to-speech model.

The caller streams microphone audio and hears the assistant answer as it
speaks; the transcript of both sides comes back alongside. The app is a relay
with a contract on each side: the platform socket, declared by the models
below, and Grok's realtime WebSocket (``wss://api.x.ai/v1/realtime``), which
carries raw PCM frames both ways once the session is configured for binary
transport, so audio passes through untouched.

    caller -> app   <binary PCM s16le mono 24 kHz>   an item of TalkInput.audio (a frame every ~20 ms)
    caller -> app   {"events": {"type": "text", "text": "hi"}}   a typed message instead of speech
    caller -> app   {"voice": "ara"}                 change an ordinary input mid-stream (a session.update)
    app -> caller   {"user_text": ""}                the first frame: Grok has accepted the session
    app -> caller   {"user_text": "what's the"}      what Grok hears, refined as the user speaks
    app -> caller   {"assistant_text": "The wea"}    what the assistant is saying, as it says it
    app -> caller   <binary PCM s16le mono 24 kHz>   an item of TalkOutput.audio
    caller closes  ->  the Grok session ends and the function yields its result

The socket is the low-latency channel; the yields are the task's output. Each
completed turn yields a cumulative snapshot of the conversation, and the yield
after the socket closes is the result and carries ``output_meta``.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from inferencesh import (
    AudioMeta,
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    Live,
    OutputMeta,
    PCM16,
    Socket,
    Stream,
    TextMeta,
)

SAMPLE_RATE = 24000
REALTIME_URL = "wss://api.x.ai/v1/realtime"

# Grok's names for the assistant's spoken text, the OpenAI realtime names it
# is compatible with, and the text-only form.
ASSISTANT_DELTA_EVENTS = {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
}
ASSISTANT_DONE_EVENTS = {
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
    "response.output_text.done",
    "response.text.done",
}
USER_TRANSCRIPT_EVENTS = {
    "conversation.item.input_audio_transcription.updated",
    "conversation.item.input_audio_transcription.completed",
}
# Lifecycle chatter the app has no use for.
QUIET_EVENTS = {
    "session.created",
    "conversation.created",
    "conversation.item.added",
    "conversation.item.created",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "ping",
}


# xAI's roster as of September 2026 (GET /v1/tts/voices); `voices` reads the live list.
BuiltInVoice = Literal[
    "eve", "ara", "rex", "sal", "leo",
    "altair", "atlas", "aurora", "carina", "castor", "celeste", "cosmo", "helios", "helix", "iris", "kepler",
    "liora", "lumen", "luna", "lux", "naksh", "orion", "perseus", "rigel", "sirius", "ursa", "zagan", "zenith",
]


class UserText(BaseModel):
    """A typed user message. The assistant answers it out loud."""

    type: Literal["text"] = "text"
    text: str = Field(description="What the user says")


class Say(BaseModel):
    """Words the assistant speaks verbatim, as a turn of its own."""

    type: Literal["say"] = "say"
    text: str = Field(description="What the assistant says, word for word")
    interruptible: bool = Field(default=True, description="False drops the caller's audio until it has been said")


class TalkInput(BaseAppInput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")
    events: Stream[Union[UserText, Say]] = Field(description="Typed messages: text from the user, or words for the assistant")
    instructions: str = Field(
        default="You are a helpful assistant.",
        description="System prompt. Grok's voice models take plain instructions; workarounds written for other models are unnecessary.",
    )
    voice: BuiltInVoice = Field(default="eve", description="A built-in voice; `voices` lists them with what xAI says about each")
    custom_voice: Optional[str] = Field(
        default=None,
        description="The id of a voice cloned with xAI's Custom Voices API. Set, it is used instead of voice.",
    )
    model: str = Field(
        default="grok-voice-latest",
        description="grok-voice-latest follows the newest model; pin a versioned name such as grok-voice-think-fast-2.0 for stability",
    )
    reasoning: Literal["high", "none"] = Field(
        default="high",
        description="Whether the model thinks before it answers. 'none' answers faster.",
    )
    language: Optional[str] = Field(
        default=None,
        description="BCP-47 hint for what the user speaks, such as en, ja or es-MX (Spanish and Portuguese need a region). Left empty it is detected.",
    )
    speed: float = Field(default=1.0, ge=0.7, le=1.5, description="Playback speed of the assistant's voice")
    idle_minutes: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="End the session after this long with nobody speaking or typing (0: never). Grok bills every minute the session is open, silent or not.",
    )
    silence_ms: Optional[int] = Field(
        default=None, ge=0, le=10000, description="Silence that ends the user's turn, in ms. Left empty Grok decides."
    )
    web_search: bool = Field(default=False, description="Let the assistant search the web")
    x_search: bool = Field(default=False, description="Let the assistant search X")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class TalkOutput(BaseAppOutput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="The assistant's voice")
    user_text: str = Field(default="", description="What Grok hears the user saying, refined as they speak")
    assistant_text: str = Field(default="", description="What the assistant is saying, as it says it")
    messages: List[Message] = Field(default_factory=list, description="The conversation so far, a message per turn")
    seconds: float = Field(default=0, description="How long the session has run")
    partial: bool = Field(default=True, description="True while the conversation goes on; False on the result")
    end_reason: str = Field(default="", description="Why the session ended, on the result: the caller closed it, or Grok did (for example after 15 minutes without audio)")


class VoicesInput(BaseAppInput):
    pass


class Voice(BaseModel):
    id: str = Field(description="What to pass as voice or custom_voice")
    name: str = Field(default="", description="Display name")
    description: str = Field(default="", description="Tone and character, as xAI describes it")
    custom: bool = Field(default=False, description="A voice cloned by this account")


class VoicesOutput(BaseAppOutput):
    voices: List[Voice] = Field(description="Built-in voices, then this account's custom voices")


# Ordinary input fields that change the Grok session when the caller changes them mid-stream.
SESSION_FIELDS = {
    "instructions", "voice", "custom_voice", "reasoning", "language", "speed", "silence_ms", "web_search", "x_search",
}
VOICES_URL = "https://api.x.ai/v1/tts/voices"

_DONE = object()


class _Ended:
    """Grok closed a session it had accepted: the conversation so far is the result."""

    def __init__(self, reason: str):
        self.reason = reason


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY environment variable is required")
        self._socket: Optional[Socket] = None

    async def on_cancel(self):
        # Stop reading from the caller: the uplink loop ends and the session winds down.
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close()
        return True

    async def voices(self, input_data: VoicesInput) -> VoicesOutput:
        """The voices `talk` can use: xAI's built-in roster and this account's custom voices."""
        import urllib.request

        def fetch() -> Any:
            req = urllib.request.Request(VOICES_URL, headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read())

        data = await asyncio.to_thread(fetch)
        items = data.get("voices") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise RuntimeError(f"unexpected voices response: {json.dumps(data)[:300]}")
        voices = []
        for item in items:
            if not isinstance(item, dict):
                continue
            voice_id = item.get("voice_id") or item.get("id") or item.get("name")
            if not voice_id:
                continue
            voices.append(
                Voice(
                    id=str(voice_id),
                    name=str(item.get("name") or item.get("display_name") or voice_id),
                    description=str(item.get("description") or ""),
                    custom=bool(item.get("custom") or item.get("is_custom") or item.get("type") == "custom"),
                )
            )
        self.logger.info("voices: %d (keys of the first: %s)", len(voices), list(items[0].keys()) if items else [])
        return VoicesOutput(voices=voices)

    async def talk(self, input_data: TalkInput, socket: Socket) -> AsyncGenerator[TalkOutput, None]:
        import websockets

        live = Live(socket, input_data, TalkOutput)
        self._socket = socket
        conversation = _Conversation()
        ready = asyncio.Event()                 # Grok accepted the session
        snapshots: "asyncio.Queue[Any]" = asyncio.Queue()
        started = time.monotonic()
        audio_in = audio_out = 0                # bytes
        text_inputs = 0

        grok = await websockets.connect(
            f"{REALTIME_URL}?model={input_data.model}",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )
        await grok.send(json.dumps({"type": "session.update", "session": _session(input_data)}))

        async def uplink() -> None:
            nonlocal audio_in, text_inputs
            await ready.wait()
            async for update in live:
                if update.field == "audio":
                    audio_in += len(update.value)
                    await grok.send(update.value)
                elif update.field == "events":
                    activity["at"] = time.monotonic()
                    text_inputs += 1
                    await grok.send(json.dumps(_item(update.value)))
                    if isinstance(update.value, UserText):
                        # Grok transcribes nothing for a typed turn, so place it here.
                        conversation.turn_of_user()
                        conversation.messages.append(Message(role="user", text=update.value.text))
                        snapshots.put_nowait(None)
                        await grok.send(json.dumps({"type": "response.create"}))
                elif update.field in SESSION_FIELDS:
                    await grok.send(json.dumps({"type": "session.update", "session": _session(input_data)}))
            snapshots.put_nowait(_DONE)

        async def downlink() -> None:
            nonlocal audio_out
            try:
                async for message in grok:
                    if isinstance(message, bytes):
                        audio_out += len(message)
                        await live.send(audio=message)
                        continue
                    event = json.loads(message)
                    kind = event.get("type", "")
                    if kind == "session.updated":
                        if not ready.is_set():
                            ready.set()
                            await live.send(user_text="")     # the first frame: the app is there
                    elif kind in USER_TRANSCRIPT_EVENTS:
                        # Cumulative per item, and Grok repeats it (also empty,
                        # once a second, while the user is silent).
                        text = event.get("content") or event.get("transcript") or ""
                        if conversation.heard(event.get("item_id"), text):
                            await live.send(user_text=conversation.user_text)
                    elif kind == "input_audio_buffer.speech_started":
                        activity["at"] = time.monotonic()
                        # The user talked over the answer: Grok stops, but the
                        # caller may have seconds of it queued. Drop them.
                        if audio_out > activity["cleared_at"]:
                            activity["cleared_at"] = audio_out
                            await live.clear("audio")
                    elif kind == "response.created":
                        activity["responding"] = True
                        if conversation.turn_of_user():
                            snapshots.put_nowait(None)
                        await live.send(assistant_text="")
                    elif kind in ASSISTANT_DELTA_EVENTS:
                        conversation.assistant_text += event.get("delta") or ""
                        await live.send(assistant_text=conversation.assistant_text)
                    elif kind in ASSISTANT_DONE_EVENTS:
                        text = event.get("transcript") or event.get("text")
                        if text:
                            conversation.assistant_text = text
                            await live.send(assistant_text=text)
                    elif kind == "response.done":
                        activity["responding"] = False
                        activity["at"] = time.monotonic()
                        usage = (event.get("response") or {}).get("usage")
                        if usage:
                            self.logger.info("turn usage: %s", json.dumps(usage))
                        if conversation.turn_of_assistant():
                            snapshots.put_nowait(None)
                    elif kind == "error":
                        # Not always fatal (a rejected setting leaves the session
                        # up); Grok closes the socket when it is. Tell the caller.
                        err = event.get("error") or {}
                        grok_error["message"] = f"Grok: {err.get('message') or json.dumps(err)}"
                        self.logger.warning("%s", grok_error["message"])
                        await live.error(grok_error["message"])
                    elif kind not in QUIET_EVENTS:
                        self.logger.info("grok event %s: %s", kind, message[:300])
            except asyncio.CancelledError:
                raise
            except Exception as err:
                grok_error.setdefault("message", f"Grok closed the session: {err}")
            if uplink_task.done():
                return                              # the caller closed first; the result is on its way
            # Grok is gone. The caller's loop is blocked on the socket, so say
            # so through the snapshot queue. A session Grok never accepted is
            # a failure; one it accepted ends like any other and is billed.
            reason = grok_error.get("message") or "Grok closed the session"
            snapshots.put_nowait(_Ended(reason) if ready.is_set() else RuntimeError(reason))

        async def idle_watch() -> None:
            # Silence still streams, so Grok does not see an idle session;
            # nobody speaking or typing (and Grok not answering) is idle.
            while True:
                limit = input_data.idle_minutes * 60      # read each time: it can change mid-stream
                await asyncio.sleep(min(5.0, limit / 4) if limit > 0 else 5.0)
                if limit <= 0 or activity["responding"] or time.monotonic() - activity["at"] < limit:
                    continue
                minutes = f"{input_data.idle_minutes:g} minute{'' if input_data.idle_minutes == 1 else 's'}"
                reason = f"ended after {minutes} with nobody speaking"
                self.logger.info("%s", reason)
                await live.error(reason)
                snapshots.put_nowait(_Ended(reason))
                return

        def snapshot(partial: bool = True) -> TalkOutput:
            return TalkOutput(
                user_text=conversation.user_text,
                assistant_text=conversation.assistant_text,
                messages=list(conversation.messages),
                seconds=round(time.monotonic() - started, 3),
                partial=partial,
            )

        grok_error: Dict[str, str] = {}
        activity: Dict[str, Any] = {"at": time.monotonic(), "responding": False, "cleared_at": 0}
        end_reason = "the caller closed the session"
        uplink_task = asyncio.create_task(uplink())
        downlink_task = asyncio.create_task(downlink())
        idle_task = asyncio.create_task(idle_watch())
        try:
            while True:
                item = await snapshots.get()
                if item is _DONE:
                    break
                if isinstance(item, _Ended):
                    end_reason = item.reason
                    break
                if isinstance(item, BaseException):
                    raise item
                yield snapshot()
        finally:
            self._socket = None
            for task in (uplink_task, downlink_task, idle_task):
                task.cancel()
            await asyncio.gather(uplink_task, downlink_task, idle_task, return_exceptions=True)
            await grok.close()

        # The caller is gone: whatever was mid-turn is the last of the conversation.
        conversation.turn_of_user()
        conversation.turn_of_assistant()
        seconds = time.monotonic() - started
        result = snapshot(partial=False)
        result.end_reason = end_reason
        # What xAI bills: the session's minutes as one audio input, and each
        # typed message as one text input. The assistant's speech is reported
        # for the record.
        result.output_meta = OutputMeta(
            inputs=[
                AudioMeta(
                    seconds=round(seconds, 3),
                    sample_rate=SAMPLE_RATE,
                    extra={
                        "model": input_data.model,
                        "voice": input_data.custom_voice or input_data.voice,
                        "audio_in_seconds": round(audio_in / (SAMPLE_RATE * 2), 3),
                    },
                ),
                *[TextMeta() for _ in range(text_inputs)],
            ],
            outputs=[AudioMeta(seconds=round(audio_out / (SAMPLE_RATE * 2), 3), sample_rate=SAMPLE_RATE)],
        )
        self.logger.info(
            "session ended (%s): %.1fs, %d turns, %.1fs of audio in, %.1fs out",
            end_reason, seconds, len(conversation.messages), audio_in / (SAMPLE_RATE * 2), audio_out / (SAMPLE_RATE * 2),
        )
        yield result


class _Conversation:
    """The transcript as Grok reports it: the user's side arrives as cumulative
    updates per item (corrections included), the assistant's as deltas."""

    def __init__(self) -> None:
        self.messages: List[Message] = []
        self.user_text = ""
        self.assistant_text = ""
        self._user_item: Optional[str] = None
        self._placed: Dict[str, int] = {}       # item id -> index in messages, once the turn is placed

    def heard(self, item_id: Optional[str], text: str) -> bool:
        """Applies what Grok heard; True when the user's live text changed."""
        if item_id is not None and item_id in self._placed:
            self.messages[self._placed[item_id]].text = text   # a correction to a placed turn
            return False
        changed = text != self.user_text
        self._user_item = item_id
        self.user_text = text
        return changed

    def turn_of_user(self) -> bool:
        if not self.user_text:
            return False
        self.messages.append(Message(role="user", text=self.user_text))
        if self._user_item is not None:
            self._placed[self._user_item] = len(self.messages) - 1
        self.user_text, self._user_item = "", None
        return True

    def turn_of_assistant(self) -> bool:
        if not self.assistant_text:
            return False
        self.messages.append(Message(role="assistant", text=self.assistant_text))
        self.assistant_text = ""
        return True


def _session(input_data: TalkInput) -> Dict[str, Any]:
    turn_detection: Dict[str, Any] = {"type": "server_vad"}
    if input_data.silence_ms is not None:
        turn_detection["silence_duration_ms"] = input_data.silence_ms
    transcription: Dict[str, Any] = {"model": "grok-transcribe"}   # asks for the user's transcript
    if input_data.language:
        transcription["language_hint"] = input_data.language
    tools = []
    if input_data.web_search:
        tools.append({"type": "web_search"})
    if input_data.x_search:
        tools.append({"type": "x_search"})
    return {
        "instructions": input_data.instructions,
        "voice": input_data.custom_voice or input_data.voice,
        "reasoning": {"effort": input_data.reasoning},
        "turn_detection": turn_detection,
        "tools": tools,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transport": "binary",
                "transcription": transcription,
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transport": "binary",
                "speed": input_data.speed,
            },
        },
    }


def _item(event: Union[UserText, Say]) -> Dict[str, Any]:
    if isinstance(event, Say):
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "force_message",
                "role": "assistant",
                "interruptible": event.interruptible,
                "content": [{"type": "output_text", "text": event.text}],
            },
        }
    return {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": event.text}]},
    }
