# ElevenLabs API Reference

API Key: `ELEVENLABS_KEY`
Base URL: `https://api.elevenlabs.io/v1`

## Authentication

```
xi-api-key: YOUR_API_KEY
```

---

## Text to Speech

**POST** `/text-to-speech/{voice_id}`

Request body:
```json
{
  "text": "Text to convert",
  "model_id": "eleven_multilingual_v2",
  "output_format": "mp3_44100_128",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": true
  }
}
```

Models:
- `eleven_multilingual_v2` - Highest quality, 32 languages, ~250-300ms latency
- `eleven_turbo_v2_5` - Lower latency
- `eleven_flash_v2_5` - Ultra-low latency (~75ms)

Output formats:
- `mp3_44100_128`, `mp3_44100_192`
- `pcm_16000`, `pcm_22050`, `pcm_24000`, `pcm_44100`

Response: Audio stream

---

## Speech to Text

**POST** `/speech-to-text`

Multipart form data:
- `file`: Audio file
- `model_id`: `scribe_v1` or `scribe_v2`
- `language_code`: e.g., "eng", "spa" (optional, auto-detect if omitted)
- `diarize`: true/false
- `tag_audio_events`: true/false

Response:
```json
{
  "text": "Transcribed text...",
  "language_code": "eng",
  "language_probability": 0.98,
  "words": [
    {"text": "Hello", "start": 0.0, "end": 0.5, "speaker": "speaker_0"}
  ]
}
```

---

## Voice Changer (Speech to Speech)

**POST** `/speech-to-speech/{voice_id}`

Multipart form data:
- `audio`: Input audio file
- `model_id`: `eleven_multilingual_sts_v2` or `eleven_english_sts_v2`
- `output_format`: Same as TTS

Response: Audio stream

---

## Voice Isolator

**POST** `/audio-isolation`

Multipart form data:
- `audio`: Input audio file (max 500MB / 1 hour)

Response: Audio stream (isolated voice)

---

## Sound Effects

**POST** `/sound-generation`

Request body:
```json
{
  "text": "Cinematic Braam, Horror",
  "duration_seconds": 5.0,
  "prompt_influence": 0.3
}
```

- `duration_seconds`: 0.5-22 seconds (optional)
- `prompt_influence`: 0-1, how literal to interpret prompt

Response: Audio stream

---

## Music Generation

**POST** `/music/compose`

Request body:
```json
{
  "prompt": "Upbeat electronic track with driving synths...",
  "music_length_ms": 30000
}
```

- `music_length_ms`: 5000-600000 (5 seconds to 10 minutes)

Alternative: `composition_plan` instead of `prompt` for structured control.

Response: Audio stream

---

## Text to Dialogue

**POST** `/text-to-dialogue/convert`

Request body:
```json
{
  "script": [
    {
      "voice_id": "JBFqnCBsd6RMkjVDRZzb",
      "text": "Hello, how are you?"
    },
    {
      "voice_id": "9BWtsMINqrJLrRacOk9x",
      "text": "I'm doing great, thanks for asking!"
    }
  ],
  "model_id": "eleven_multilingual_v2",
  "output_format": "mp3_44100_128"
}
```

- Script: Array of voice/text pairs for multi-speaker dialogue
- model_id: Same as TTS models
- output_format: Same as TTS formats

Response: Audio stream (single file with all dialogue)

---

## Forced Alignment

**POST** `/forced-alignment`

Multipart form data:
- `audio`: Audio file
- `text`: Text to align with audio

Response:
```json
{
  "alignment": {
    "words": [
      {"text": "Hello", "start": 0.0, "end": 0.5},
      {"text": "world", "start": 0.6, "end": 1.0}
    ],
    "characters": [
      {"text": "H", "start": 0.0, "end": 0.1},
      ...
    ]
  }
}
```

- Provides precise timestamps for words and characters
- Useful for subtitles, lip-sync, karaoke
- Returns both word-level and character-level alignment

---

## Dubbing

### Create Dubbing

**POST** `/dubbing`

Multipart form data:
- `file`: Audio/video file
- `target_lang`: Target language code (es, fr, de, etc.)
- `source_lang`: Source language (optional, auto-detect)

Response:
```json
{
  "dubbing_id": "xxx-xxx-xxx"
}
```

### Check Status

**GET** `/dubbing/{dubbing_id}`

Response:
```json
{
  "status": "dubbing" | "dubbed" | "failed",
  "error": "..." // if failed
}
```

### Get Dubbed Audio

**GET** `/dubbing/{dubbing_id}/audio/{language_code}`

Response: Audio stream

---

## Voices

### List Voices

**GET** `/voices`

Response:
```json
{
  "voices": [
    {
      "voice_id": "JBFqnCBsd6RMkjVDRZzb",
      "name": "George",
      "category": "premade",
      "labels": {"accent": "british", "gender": "male"}
    }
  ]
}
```

### Default Voice IDs

| Name | Voice ID | Description |
|------|----------|-------------|
| George | JBFqnCBsd6RMkjVDRZzb | British male |
| Aria | 9BWtsMINqrJLrRacOk9x | American female |
| Charlie | IKne3meq5aSn9XLyUdCD | Australian male |

---

## Supported Languages (TTS)

32 languages including: English, Spanish, French, German, Italian, Portuguese, Polish, Hindi, Arabic, Chinese, Japanese, Korean, Russian, Turkish, Dutch, Swedish, Danish, Finnish, Norwegian, Czech, Greek, Hebrew, Hungarian, Indonesian, Malay, Romanian, Thai, Ukrainian, Vietnamese.

---

## Error Handling

Status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `422`: Unprocessable entity (validation error)
- `429`: Rate limited
- `500`: Server error

Error response:
```json
{
  "detail": {
    "status": "error_code",
    "message": "Human readable error"
  }
}
```
