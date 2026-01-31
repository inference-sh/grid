# Kling AI - Lip Sync API

Drive mouth shape of characters in videos using text or audio.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/lip-sync` |
| Query Single | GET | `/v1/videos/lip-sync/{task_id}` |
| Query List | GET | `/v1/videos/lip-sync?pageNum=1&pageSize=30` |

---

## Description

The Lip Sync API modifies existing videos to make characters' lips match new audio or text. This enables:
- Dubbing videos in different languages
- Adding speech to silent videos
- Replacing dialogue in videos
- Creating talking character content

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_url` | string | **Yes** | - | Source video URL |
| `text` | string | Conditional | - | Text for lip sync |
| `audio_url` | string | Conditional | - | Audio for lip sync |
| `voice_id` | string | Conditional | - | TTS voice ID (if using text) |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

**Note:** Either `text` + `voice_id` OR `audio_url` must be provided.

---

## Video Requirements

- Clear face visible
- Good lighting
- Relatively stable head position
- Formats: `.mp4`, `.mov`
- Duration appropriate for content

---

## Examples

### With Text and Voice

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/lip-sync' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "video_url": "https://example.com/source-video.mp4",
    "text": "This is the new dialogue I want the character to speak.",
    "voice_id": "voice-id-here"
}'
```

### With Audio

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/lip-sync' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "video_url": "https://example.com/source-video.mp4",
    "audio_url": "https://example.com/new-dialogue.mp3"
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

## Combining with Video Generation

You can use lip sync with Kling-generated videos:

1. Generate video with Image to Video
2. Get the video URL from result
3. Apply lip sync with new audio/text

```bash
# Step 1: Generate video
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "image": "https://example.com/person.jpg",
    "prompt": "Person standing calmly, looking at camera",
    "mode": "pro"
}'

# Step 2: Apply lip sync to result
curl -X POST 'https://api-singapore.klingai.com/v1/videos/lip-sync' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "video_url": "https://result-from-step-1.mp4",
    "text": "Hello, I am an AI-generated character.",
    "voice_id": "voice-id-here"
}'
```

---

## Gotchas

1. **Clear face required** - Face must be visible for lip sync to work
2. **Audio/video length** - Output duration matches the audio duration
3. **Head stability** - Works best with relatively stable head position
4. **Either text+voice or audio** - Can't use both simultaneously
5. **Voice ID required for text** - Must specify TTS voice when using text input
