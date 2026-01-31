# Kling AI - Custom Voice API

Create personalized voice clones for use with TTS and video generation.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Voice | POST | `/v1/voices/clone` |
| List Voices | GET | `/v1/voices` |
| Delete Voice | DELETE | `/v1/voices/{voice_id}` |

---

## Description

The Custom Voice API allows you to:
- Clone voices from audio samples
- Create personalized voice profiles
- Use custom voices in TTS and video generation

---

## Voice Cloning

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_url` | string | **Yes** | Sample audio URL |
| `name` | string | **Yes** | Voice name for reference |
| `description` | string | No | Voice description |

### Audio Requirements

- Clear speech sample
- Minimal background noise
- Duration: 10-60 seconds recommended
- Formats: `.mp3`, `.wav`, `.m4a`
- Single speaker

---

## Example

### Clone a Voice

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/voices/clone' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "audio_url": "https://example.com/voice-sample.mp3",
    "name": "Custom Voice 1",
    "description": "Male voice, professional tone"
}'
```

### List Voices

```bash
curl -X GET 'https://api-singapore.klingai.com/v1/voices' \
-H 'Authorization: Bearer xxx'
```

### Delete Voice

```bash
curl -X DELETE 'https://api-singapore.klingai.com/v1/voices/voice-id-here' \
-H 'Authorization: Bearer xxx'
```

---

## Response

### Clone Response

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "voice_id": "string",
    "name": "string",
    "status": "ready",
    "created_at": 1722769557708
  }
}
```

### List Response

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "voices": [
      {
        "voice_id": "string",
        "name": "string",
        "description": "string",
        "status": "ready",
        "created_at": 1722769557708
      }
    ]
  }
}
```

---

## Using Custom Voices

### With TTS

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/audio/tts' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "text": "Hello, this is my cloned voice speaking.",
    "voice_id": "custom-voice-id-from-clone"
}'
```

### With Image to Video (V2.6)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v2-6",
    "image": "https://example.com/person.jpg",
    "prompt": "The person <<<voice_1>>> says \"Hello world\"",
    "voice_list": [{"voice_id": "custom-voice-id"}],
    "sound": "on",
    "mode": "pro"
}'
```

---

## Voice ID Sources

| Source | Description | API |
|--------|-------------|-----|
| Custom Voice | Cloned from audio | `/v1/voices/clone` |
| Official Presets | Platform-provided | Listed in `/v1/voices` |

**Note:** Voice IDs from this API are different from Lip Sync API voice IDs.

---

## Gotchas

1. **Audio quality matters** - Clear samples produce better clones
2. **Single speaker** - Multi-speaker audio will produce poor results
3. **Processing time** - Voice cloning may take time to complete
4. **Voice limit** - Account may have limit on custom voices
5. **Different from Lip Sync** - Voice IDs here are for TTS/video, not Lip Sync
