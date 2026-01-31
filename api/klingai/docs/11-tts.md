# Kling AI - Text to Speech (TTS) API

Generate speech audio from text with various voice options.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/audio/tts` |
| Query Single | GET | `/v1/audio/tts/{task_id}` |
| Query List | GET | `/v1/audio/tts?pageNum=1&pageSize=30` |

---

## Description

The TTS API converts text to natural-sounding speech. Used for:
- Generating voiceovers
- Creating audio for Avatar/Lip Sync
- Producing podcast-style content
- Accessibility features

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | **Yes** | - | Text to convert to speech |
| `voice_id` | string | **Yes** | - | Voice to use |
| `speed` | float | No | `1.0` | Speech speed multiplier |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

---

## Voice IDs

Voice IDs are obtained through:
1. Official preset voices (platform-provided)
2. Custom voice cloning API

See Custom Voice API for creating personalized voices.

---

## Example

### Basic TTS

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/audio/tts' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "text": "Hello, this is a test of the text to speech system.",
    "voice_id": "preset-voice-id"
}'
```

### With Speed Adjustment

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/audio/tts' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "text": "This will be spoken a bit faster than normal.",
    "voice_id": "preset-voice-id",
    "speed": 1.2
}'
```

---

## Response

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "task_id": "string",
    "task_status": "succeed",
    "task_status_msg": "string",
    "task_info": {"external_task_id": "string"},
    "final_unit_deduction": "string",
    "created_at": 1722769557708,
    "updated_at": 1722769557708,
    "task_result": {
      "audio": {
        "url": "https://...",
        "duration": "string"
      }
    }
  }
}
```

---

## Integration with Video

TTS output can be used with:
- **Avatar API**: Generate talking head videos
- **Lip Sync API**: Add speech to existing videos
- **Image to Video (V2.6)**: Voice control with `<<<voice_1>>>`

### Workflow Example

```bash
# Step 1: Generate speech
curl -X POST 'https://api-singapore.klingai.com/v1/audio/tts' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "text": "Welcome to our product demonstration.",
    "voice_id": "voice-id"
}'

# Step 2: Use audio with Lip Sync
curl -X POST 'https://api-singapore.klingai.com/v1/videos/lip-sync' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "video_url": "https://example.com/video.mp4",
    "audio_url": "https://tts-result-url.mp3"
}'
```

---

## Gotchas

1. **Voice ID required** - Must specify a valid voice ID
2. **Text length limits** - Very long text may need to be split
3. **Speed range** - Typical range is 0.5 to 2.0
4. **Output format** - Returns audio URL (typically MP3)
