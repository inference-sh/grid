# grok-voice

Talk with Grok. Stream microphone audio in and xAI's speech-to-speech model
answers out loud as it speaks, with both sides transcribed. When the caller
closes the socket the Grok session ends and the conversation is the task's
result.

The app is a relay between two contracts: the platform socket, declared by the
input and output models, and Grok's realtime WebSocket
(`wss://api.x.ai/v1/realtime`). The Grok session is configured for binary
transport, so PCM frames pass through in both directions without re-encoding.

## What the socket carries

`audio` is a live field on both sides, so mic frames and the assistant's voice
are binary frames. `events` is a live field of typed messages. `user_text` and
`assistant_text` are ordinary output fields, so they come back as JSON patches.

```
caller -> app   <binary PCM s16le mono 24 kHz>              an item of audio (a frame every ~20 ms)
caller -> app   {"events": {"type": "text", "text": "hi"}}  a typed user message; the assistant answers it
caller -> app   {"events": {"type": "say", "text": "..."}}  words the assistant speaks verbatim
caller -> app   {"voice": "ara"}                            change an ordinary input mid-stream
app -> caller   {"user_text": ""}                           the first frame: Grok accepted the session
app -> caller   {"user_text": "what's the"}                 what Grok hears, refined as the user speaks
app -> caller   {"assistant_text": "The wea"}               what the assistant is saying, as it says it
app -> caller   <binary PCM s16le mono 24 kHz>              an item of audio: the assistant's voice
caller closes  ->  the function yields its result
```

Turns are detected by Grok's server VAD. When the user talks over the
assistant Grok stops sending audio; what the caller has already buffered still
plays out.

## What the yields are

`talk` is an async generator: every completed turn is a progress snapshot of the
task's output, reaching the caller over SSE and the task page. Snapshots are
cumulative; the last one is the result.

| field | |
|---|---|
| `messages` | the conversation so far, `{role, text}` per turn |
| `user_text` | the user's turn in progress |
| `assistant_text` | the assistant's turn in progress |
| `seconds` | how long the session has run |
| `partial` | `true` while it goes on, `false` on the result |

The result's `output_meta` carries what xAI bills: the session length as one
audio input (per minute) and each typed message as one text input. The
assistant's speech is reported as audio seconds out.

## Input

| field | |
|---|---|
| `audio` | live: mic frames, PCM s16le mono 24 kHz |
| `events` | live: `text` (a typed user message) or `say` (verbatim words for the assistant) |
| `instructions` | system prompt |
| `voice` | one of xAI's 28 built-in voices, `eve` by default |
| `custom_voice` | the id of a voice cloned with xAI's Custom Voices API; set, it replaces `voice` |
| `model` | `grok-voice-latest` (default) or a pinned name such as `grok-voice-think-fast-2.0` |
| `reasoning` | `high` (default) or `none` for faster answers |
| `language` | BCP-47 hint for the user's language; Spanish and Portuguese need a region (`es-MX`, `pt-BR`) |
| `speed` | playback speed of the voice, 0.7 to 1.5 |
| `silence_ms` | silence that ends the user's turn; left empty Grok decides |
| `web_search`, `x_search` | tools the assistant may use |

Every ordinary field except `model` can be changed while the stream runs; the
app sends Grok a `session.update`.

## voices

`voices` (no input) lists the voices `talk` can use: xAI's built-in roster and
this account's custom voices, as `{id, name, description, custom}`. A free,
read-only call.

## Calling it

```python
import asyncio
from inferencesh import async_inference

async def main():
    client = async_inference(api_key="your-api-key")

    task, session = await client.live(
        {"app": "xai/grok-voice", "function": "talk",
         "input": {"instructions": "You are a terse assistant.", "voice": "eve"}},
        on_state=lambda state, end: print(state),   # connecting → waiting → live → ended
    )

    async def read():
        async for frame in session:
            if isinstance(frame, bytes):
                play(frame)                          # 24 kHz s16le
            else:
                print(frame)                         # {"user_text": ...} / {"assistant_text": ...}

    reader = asyncio.create_task(read())
    for frame in mic_frames():                       # 960 bytes: 20 ms of 24 kHz s16le
        await session.send(frame)
    await session.send({"events": {"type": "text", "text": "Goodbye."}})
    await asyncio.sleep(5)

    await session.close()
    reader.cancel()
    print((await client.tasks.get(task["id"]))["output"]["messages"])

asyncio.run(main())
```

The socket needs the async extra: `pip install inferencesh[async]`. The session
is `waiting` until Grok accepts the session, which takes a cold start plus a
round trip to xAI.

## Locally

```bash
XAI_API_KEY=... belt app test --stream --function talk --stream-addr 127.0.0.1:8765
```

Open the printed URL and talk into the microphone. Use headphones: the
assistant's voice plays while the microphone is open.
