# Kling AI API Reference

Base URL: `https://api-singapore.klingai.com`
Auth: JWT (HS256) with AccessKey/SecretKey

## Video Generation

### Text to Video

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/text2video` |
| GET | `/v1/videos/text2video/{task_id}` |
| GET | `/v1/videos/text2video?pageNum=1&pageSize=30` |

**Models:** `kling-v1`, `kling-v1-6`, `kling-v2-master`, `kling-v2-1-master`, `kling-v2-5-turbo`, `kling-v2-6`

| Model | Modes | Resolution | FPS | Sound | Duration |
|-------|-------|-----------|-----|-------|----------|
| kling-v1 | std, pro | 720p | 30 | - | 5s, 10s |
| kling-v1-6 | std, pro | 720p (std) / 1080p (pro) | 30 | - | 5s, 10s |
| kling-v2-master | - | 720p | 24 | - | 5s, 10s |
| kling-v2-1-master | - | 1080p | 24 | - | 5s, 10s |
| kling-v2-5-turbo | std, pro | 720p (std) / 1080p (pro) | 24 | - | 5s, 10s |
| kling-v2-6 | std, pro | 720p (std) / 1080p (pro) | 24 | pro only | 5s, 10s |

### Image to Video

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/image2video` |
| GET | `/v1/videos/image2video/{task_id}` |
| GET | `/v1/videos/image2video?pageNum=1&pageSize=30` |

**Models:** `kling-v1`, `kling-v1-5`, `kling-v1-6`, `kling-v2-master`, `kling-v2-1`, `kling-v2-1-master`, `kling-v2-5-turbo`, `kling-v2-6`

| Model | Modes | Start/End Frame | Voice | Motion Brush | Camera Control |
|-------|-------|----------------|-------|-------------|----------------|
| kling-v1 | std, pro | std/pro 5s | - | std/pro 5s | - |
| kling-v1-5 | std, pro | pro 5s/10s | - | pro 5s | simple (pro 5s) |
| kling-v1-6 | std, pro | pro 5s/10s | - | - | - |
| kling-v2-master | - | - | - | - | - |
| kling-v2-1 | std, pro | pro 5s/10s | - | - | - |
| kling-v2-1-master | - | - | - | - | - |
| kling-v2-5-turbo | std, pro | pro 5s/10s | - | - | - |
| kling-v2-6 | std, pro | pro (no audio) | pro | std (motion ctrl) | - |

### Omni-Video (O1)

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/omni-video` |
| GET | `/v1/videos/omni-video/{task_id}` |
| GET | `/v1/videos/omni-video?pageNum=1&pageSize=30` |

**Model:** `kling-video-o1` (std, pro)

Capabilities:
- Text to video (5s, 10s)
- Image/element reference (up to 7 without video, 4 with video)
- Start + end frame control
- Video reference for style/motion (`refer_type: feature`)
- Video editing/transformation (`refer_type: base`)
- Duration: 3-10s (varies by mode)

### Video Extension

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/video-extend` |
| GET | `/v1/videos/video-extend/{task_id}` |

**Models:** `kling-v1`, `kling-v1-5`, `kling-v1-6` (std, pro)

### Video Effects

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/effects` |
| GET | `/v1/videos/effects/{task_id}` |

**Models:** `kling-v1`, `kling-v1-5`, `kling-v1-6` (std, pro)
**Effects:** `hug`, `kiss`, `heart_gesture` (dual-character)

---

## Image Generation

| Method | Endpoint |
|--------|----------|
| POST | `/v1/images/generations` |
| GET | `/v1/images/generations/{task_id}` |
| GET | `/v1/images/generations?pageNum=1&pageSize=30` |

**Models:**

| Model | Text-to-Image | Image-to-Image | Resolution | Ratios |
|-------|--------------|----------------|------------|--------|
| kling-v1 | yes | entire image | 1K | 1:1, 16:9, 4:3, 3:2, 2:3, 3:4, 9:16 |
| kling-v1-5 | yes | subject, face | 1K | + 21:9 |
| kling-v2 | yes | multi-image, restyle | 1K/2K (t2i) | all |
| kling-v2-new | - | restyle only | input res | all |
| kling-v2-1 | yes | multi-image | 1K | all |
| kling-image-o1 | yes | yes | - | all + auto |

### Image Expansion

| Method | Endpoint |
|--------|----------|
| POST | `/v1/images/image-expansion` |
| GET | `/v1/images/image-expansion/{task_id}` |

---

## Audio & Voice

### Video to Audio

| Method | Endpoint |
|--------|----------|
| POST | `/v1/audios/video2audio` |
| GET | `/v1/audios/video2audio/{task_id}` |

Add audio to any video (Kling-generated or uploaded).

### Text to Speech

| Method | Endpoint |
|--------|----------|
| POST | `/v1/audios/tts` |
| GET | `/v1/audios/tts/{task_id}` |

### Custom Voice

| Method | Endpoint |
|--------|----------|
| POST | `/v1/audios/voice-clone` |
| GET | `/v1/audios/voice-clone/{task_id}` |

---

## Digital Human

### Avatar (Broadcast-Style)

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/avatar` |
| GET | `/v1/videos/avatar/{task_id}` |

Generate digital human broadcast videos from a single photo.

### Lip Sync

| Method | Endpoint |
|--------|----------|
| POST | `/v1/videos/lipsync` |
| GET | `/v1/videos/lipsync/{task_id}` |

Drive mouth movements with text or audio.

---

## Elements (Character Consistency)

| Method | Endpoint |
|--------|----------|
| POST | `/v1/elements` |
| GET | `/v1/elements/{element_id}` |
| DELETE | `/v1/elements/{element_id}` |
| GET | `/v1/elements?pageNum=1&pageSize=30` |

Register elements (characters, items) for consistent use across Omni-Video generations.

---

## Virtual Try-On

| Method | Endpoint |
|--------|----------|
| POST | `/v1/images/virtual-try-on` |
| GET | `/v1/images/virtual-try-on/{task_id}` |

---

## Common Parameters

### Aspect Ratios
`16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `21:9`

### Task Statuses
`submitted` → `processing` → `succeed` / `failed`

### Image Requirements
- Formats: jpg, jpeg, png
- Max: 10MB, min: 300px
- Ratio: 1:2.5 to 2.5:1
- Base64: NO `data:` prefix

### Video Requirements (for references)
- Formats: mp4, mov
- Duration: 3-10s
- Resolution: 720-2160px
- FPS: 24-60
- Max: 200MB

### Content Retention
Generated files deleted after **30 days**.
