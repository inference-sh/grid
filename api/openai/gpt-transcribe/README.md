# gpt-transcribe

Transcribe speech with OpenAI.

| function | what it does | model | OpenAI endpoint |
| -- | -- | -- | -- |
| `run` (default) | a recording in, its transcript out | `gpt-transcribe` | `POST /v1/audio/transcriptions` |
| `realtime` | a live function: microphone audio in over a socket, the transcript back as it is spoken | `gpt-live-transcribe` | `wss://api.openai.com/v1/realtime?intent=transcription` |

Neither model returns word timings or speakers.

## What the socket of `realtime` carries

`audio` is a live input field, so mic frames are binary frames. `text` is an
ordinary output field, so it comes back as JSON patches.

```
caller -> app   <binary PCM s16le mono 24 kHz>   an item of audio (a frame every ~20 ms)
caller -> app   {"keyterms": ["AC-42"]}          change an ordinary input mid-stream (a session.update)
app -> caller   {"text": ""}                     the first frame: OpenAI accepted the session
app -> caller   {"text": "what's the wea"}       the transcript so far, its tail still growing
caller closes  ->  the last turn is committed and the function yields its result
```

`prompt`, `languages`, `keyterms` and `delay` can change mid-stream.

## Turns

`gpt-live-transcribe` writes words as it hears them and gives a turn its final
text only when the turn is committed. It detects no turns itself, so the app
does: a turn ends when the caller's audio has been quiet, or absent, for
`silence_ms` after speech (a level check on the PCM), and at 30 seconds if it
never pauses. A turn's `start` and `end` are where its audio began and ended in
the stream.

## What the yields are

`realtime` is an async generator. Every finished turn is a progress snapshot of
the task's output; the last yield is the result: the whole transcript, the
turns, and `end_reason`.

The session ends when the caller closes the socket, when nobody has spoken for
`idle_minutes`, or when OpenAI closes the session (it does after 60 minutes).
Each of those returns the transcript so far and is billed.

## Billing

`output_meta.inputs[0]` is one `AudioMeta` whose `seconds` is the audio OpenAI
was sent: the recording's length for `run` (as OpenAI reports it), and for
`realtime` every second streamed, silent or not. OpenAI charges $0.0045 per
minute for `gpt-transcribe` and $0.017 per minute for `gpt-live-transcribe`.

## Tests

`../test_gpt_transcribe.py` runs without the network: OpenAI is a scripted fake
socket.

```bash
PYTHONPATH=<sdk-py>/src python3 -m pytest api/openai/test_gpt_transcribe.py
```
