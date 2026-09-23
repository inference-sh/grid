# live-transcribe

Live dictation. Stream microphone audio in and the transcript comes back while
you are still talking: the tail of the sentence keeps being rewritten until a
pause settles it, and settled phrases never move again. When the caller closes
the socket the last window is flushed and the whole transcript is the task's
result.

faster-whisper (`large-v3-turbo` by default) over a rolling window, with the
Silero VAD that faster-whisper ships deciding when a phrase is over.

## What the socket carries

`audio` is a live field of the input, so mic frames are ordinary binary frames;
`text` is an ordinary output field, so it comes back as a JSON patch.

```
caller -> app   <binary PCM s16le mono 16 kHz>   an item of audio (a frame every ~20 ms)
caller -> app   {"language": "en"}               change an ordinary input mid-stream
app -> caller   {"text": ""}                     the first frame, so the caller knows the app is there
app -> caller   {"text": "so I just tried"}      the transcript so far, on every refresh
caller closes  ->  the last window is flushed and the function yields its result
```

Frames may be any size; 20 ms (320 samples) is what a microphone gives you.

## What the yields are

`transcribe` is an async generator, so every refresh is also a progress
snapshot of the task's output — the same text the socket carries, reaching the
caller over SSE (`GET /tasks/{id}/stream`, `stream=True`) and shown on the task
page. Snapshots are cumulative; the last one is the result.

| field | |
|---|---|
| `text` | the whole transcript so far: settled phrases plus the live tail |
| `partial` | `true` while the tail may still change, `false` on the result |
| `segments` | the settled phrases, `{text, start, end}`, seconds from the start of the stream |
| `seconds` | audio received |

The result carries `output_meta` with the audio seconds it heard.

## Input

| field | |
|---|---|
| `audio` | live: mic frames, PCM s16le mono 16 kHz |
| `language` | ISO code such as `en`; left empty the language is detected once and then pinned |
| `initial_prompt` | vocabulary hint: names and terms the transcript should prefer |
| `silence_ms` | trailing silence that settles a phrase, default 700 |

Setup takes `model` (default `large-v3-turbo`): any faster-whisper size or a
CTranslate2 model id on the Hub. It runs on CUDA float16 where there is a GPU
and CPU int8 where there is not — on CPU only the small sizes keep up with
speech.

## How it reads the window

Audio accumulates in a window since the last commit. Every 500 ms of new audio
the whole window is re-transcribed in a worker thread, at most one pass in
flight, and the result becomes the live tail. Every 100 ms the tail of the
window goes through Silero to find where speech last ended; once that is
`silence_ms` ago (or the window reaches whisper's 30 s), the window is read one
last time, appended to `segments` with absolute times, and cleared. The socket
loop never waits on a refresh, and the one commit it does wait on happens while
the caller is sending the silence that triggered it.

## Calling it

```python
import asyncio
from inferencesh import async_inference

async def main():
    client = async_inference(api_key="your-api-key")

    task, session = await client.live(
        {"app": "infsh/live-transcribe", "function": "transcribe",
         "input": {"language": "en", "silence_ms": 700}},
        on_state=lambda state, end: print(state),   # connecting → waiting → live → ended
    )

    async def follow():
        # The same transcript as a task update, which is also what the task stores.
        async with client.tasks.stream(task["id"]) as updates:
            async for update in updates:
                print("task:", (update.get("output") or {}).get("text"))

    async def read():
        async for frame in session:                 # {"text": "..."} as it refines
            print("live:", frame["text"])

    watcher = asyncio.create_task(follow())
    reader = asyncio.create_task(read())

    for frame in mic_frames():                      # 640 bytes: 20 ms of 16 kHz s16le
        await session.send(frame)

    await session.close()                           # the app flushes and the task completes
    reader.cancel()
    await watcher

    print((await client.tasks.get(task["id"]))["output"]["text"])

asyncio.run(main())
```

The socket needs the async extra: `pip install inferencesh[async]`. The session
is `waiting` until the app's first `{"text": ""}` frame, which can take a cold
start.

## Locally

```bash
belt app test --stream --function transcribe --setup '{"model": "base"}' --stream-addr 127.0.0.1:8765
```

Open the printed URL and talk into the microphone, or drive `/ws` with any
WebSocket client. Every connection is a fresh call; the yields print as
`[progress]` and the result as `[result]`.
