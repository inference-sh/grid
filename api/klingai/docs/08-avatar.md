# Kling AI - Avatar API

Generate digital human broadcast-style videos from a single photo.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/avatar` |
| Query Single | GET | `/v1/videos/avatar/{task_id}` |
| Query List | GET | `/v1/videos/avatar?pageNum=1&pageSize=30` |

---

## Description

The Avatar API creates talking head videos where a digital human speaks provided text or audio. This is ideal for:
- News-style presentations
- Educational content
- Announcements
- Virtual spokesperson content

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | string | **Yes** | - | Reference face image (URL or base64) |
| `text` | string | Conditional | - | Text for the avatar to speak |
| `audio_url` | string | Conditional | - | Audio for lip sync |
| `voice_id` | string | Conditional | - | TTS voice ID (if using text) |
| `aspect_ratio` | string | No | `16:9` | Output ratio |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

**Note:** Either `text` + `voice_id` OR `audio_url` must be provided.

---

## Image Requirements

- Single clear face
- Front-facing preferred
- Good lighting
- Neutral expression works best
- Formats: `.jpg`, `.jpeg`, `.png`
- Max size: 10MB
- Min dimensions: 300px

---

## Examples

### With Text and Voice

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/avatar' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "image": "https://example.com/face.jpg",
    "text": "Hello and welcome to our channel. Today we will be discussing AI video generation.",
    "voice_id": "voice-id-here",
    "aspect_ratio": "16:9"
}'
```

### With Audio

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/avatar' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "image": "https://example.com/face.jpg",
    "audio_url": "https://example.com/speech.mp3",
    "aspect_ratio": "16:9"
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
      "videos": [{
        "id": "string",
        "url": "https://...",
        "duration": "string"
      }]
    }
  }
}
```

---

## Gotchas

1. **Single face required** - Image must contain one clear face
2. **Front-facing best** - Profile angles may produce lower quality
3. **Either text+voice or audio** - Can't use both methods
4. **Voice ID required for text** - Must specify TTS voice when using text
5. **Audio duration** - Video length matches audio duration
