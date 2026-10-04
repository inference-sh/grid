# grok-stt

Transcribe speech with xAI's Grok Speech to Text (`grok-voice-transcribe-2.0`).

| function | what it does | xAI endpoint |
| -- | -- | -- |
| `run` (default) | a recording in, its transcript with word timings and speakers out | `POST https://api.x.ai/v1/stt` |
| `realtime` | a live function: microphone audio in over a socket, the transcript back as it is spoken | `wss://api.x.ai/v1/stt` |

## What the socket of `realtime` carries

`audio` is a live input field, so mic frames are binary frames. `text` is an
ordinary output field, so it comes back as JSON patches.

```
caller -> app   <binary PCM s16le mono 16 kHz>   an item of audio (a frame every ~20 ms)
app -> caller   {"text": ""}                     the first frame: xAI accepted the stream
app -> caller   {"text": "what's the wea"}       the transcript so far, its tail still changing
caller closes  ->  xAI flushes what it holds and the function yields its result
```

xAI reads its options from the connection URL, so `language`, `diarize`,
`keyterms`, `filler_words` and `silence_ms` are fixed once the stream is open.
`idle_minutes` can change mid-stream.

While the caller sends nothing the app sends silence. A microphone that gates
silence would otherwise leave xAI waiting for the pause that settles an
utterance.

## What the yields are

`realtime` is an async generator. Every settled utterance is a progress
snapshot of the task's output; the last yield is the result: the whole
transcript, the utterances with their times, and `end_reason`.

The session ends when the caller closes the socket, when nobody has spoken for
`idle_minutes`, or when xAI closes the stream. Each of those returns the
transcript so far and is billed.

## Billing

`output_meta.inputs[0]` is one `AudioMeta` whose `seconds` is the audio xAI
processed: the recording's length for `run`, and for `realtime` the audio sent
over the session, pauses included (xAI's own count when the stream closed
cleanly). xAI charges $0.10 per hour for `run` and $0.20 per hour for
`realtime`.

## Tests

`../test_grok_stt.py` runs without the network: xAI is a scripted fake socket.

```bash
PYTHONPATH=<sdk-py>/src python3 -m pytest api/xai/test_grok_stt.py
```
