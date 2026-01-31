# Kling AI - Image to Video API

Generate videos from reference images with optional motion control.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/image2video` |
| Query Single | GET | `/v1/videos/image2video/{task_id}` |
| Query List | GET | `/v1/videos/image2video?pageNum=1&pageSize=30` |

---

## Models

| Model | Description |
|-------|-------------|
| `kling-v1` | Base model (default) |
| `kling-v1-5` | V1.5 |
| `kling-v1-6` | V1.6 |
| `kling-v2-master` | V2 Master |
| `kling-v2-1` | V2.1 |
| `kling-v2-1-master` | V2.1 Master |
| `kling-v2-5-turbo` | V2.5 Turbo |
| `kling-v2-6` | V2.6 (with sound/voice) |

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | No | `kling-v1` | Model version |
| `image` | string | **Yes*** | - | Start frame (URL or base64) |
| `image_tail` | string | No | - | End frame image |
| `prompt` | string | No | - | Motion description (max 2500 chars) |
| `negative_prompt` | string | No | - | What to avoid |
| `voice_list` | array | No | - | Voice references (max 2) - V2.6+ |
| `sound` | string | No | `off` | `on`/`off` - V2.6+ |
| `cfg_scale` | float | No | `0.5` | Prompt adherence - **not for V2.x** |
| `mode` | string | No | `std` | `std` or `pro` |
| `static_mask` | string | No | - | Static brush mask |
| `dynamic_masks` | array | No | - | Dynamic brush configs (max 6) |
| `camera_control` | object | No | - | Camera movement |
| `duration` | string | No | `5` | `5` or `10` seconds |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

*At least one of `image` or `image_tail` required

---

## Feature Conflicts

**These are mutually exclusive:**
1. `image` + `image_tail` (start/end frames)
2. `dynamic_masks` / `static_mask` (motion brush)
3. `camera_control`

---

## Image Requirements

- Formats: `.jpg`, `.jpeg`, `.png`
- Max size: 10MB
- Min dimensions: 300px
- Aspect ratio: 1:2.5 to 2.5:1
- Base64: **No `data:` prefix**

---

## Motion Brush

### static_mask
Areas that should **remain static** (not move).

### dynamic_masks
Areas with specified motion trajectories.

```json
"dynamic_masks": [
  {
    "mask": "https://example.com/mask.png",
    "trajectories": [
      {"x": 279, "y": 219},
      {"x": 417, "y": 65}
    ]
  }
]
```

### Trajectory Rules
- Max 77 coordinates for 5-second video
- Coordinate range: [2, 77]
- **Origin: bottom-left corner** of image
- More coordinates = more precise trajectory
- First coordinate is starting point

### Mask Constraints
- Mask aspect ratio must match input image
- Static and dynamic mask resolutions must be identical
- **Only supported with kling-v1 in std/pro 5s mode**

---

## Voice Support (V2.6+)

Use `<<<voice_1>>>` in prompt to specify voice:

```json
{
  "model_name": "kling-v2-6",
  "image": "https://...",
  "prompt": "The person <<<voice_1>>> says 'Hello, welcome!'",
  "voice_list": [{"voice_id": "your-voice-id"}],
  "sound": "on",
  "mode": "pro"
}
```

- Max 2 voices per task
- Billed at "with voice generation" rate
- Simple grammar structure works best

---

## Examples

### Basic Image to Video

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "image": "https://example.com/image.jpg",
    "prompt": "The person slowly turns their head and smiles",
    "mode": "pro",
    "duration": "5"
}'
```

### Start and End Frames

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1-5",
    "image": "https://example.com/start.jpg",
    "image_tail": "https://example.com/end.jpg",
    "prompt": "Person walking from left to right",
    "mode": "pro",
    "duration": "5"
}'
```

### With Motion Brush

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "image": "https://example.com/astronaut.jpg",
    "prompt": "The astronaut stood up and walked away",
    "mode": "pro",
    "duration": "5",
    "cfg_scale": 0.5,
    "static_mask": "https://example.com/static_mask.png",
    "dynamic_masks": [{
        "mask": "https://example.com/dynamic_mask.png",
        "trajectories": [
            {"x": 279, "y": 219},
            {"x": 350, "y": 150},
            {"x": 417, "y": 65}
        ]
    }]
}'
```

### With Camera Control

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1-5",
    "image": "https://example.com/landscape.jpg",
    "prompt": "Gentle movement through the scene",
    "mode": "pro",
    "duration": "5",
    "camera_control": {
        "type": "simple",
        "config": {
            "horizontal": 3,
            "vertical": 0,
            "pan": 0,
            "tilt": 0,
            "roll": 0,
            "zoom": 0
        }
    }
}'
```

### With Voice (V2.6+)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/image2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v2-6",
    "image": "https://example.com/person.jpg",
    "prompt": "The person <<<voice_1>>> says \"Welcome to our channel!\"",
    "voice_list": [{"voice_id": "voice-id-here"}],
    "sound": "on",
    "mode": "pro",
    "duration": "5"
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
        "duration": "5"
      }]
    }
  }
}
```

---

## Model Feature Support

| Feature | v1 std 5s | v1 pro 5s | v1-5 pro 5s | v1-6 pro |
|---------|-----------|-----------|-------------|----------|
| Video generation | ✅ | ✅ | ✅ | ✅ |
| Start/end frame | ✅ | ✅ | ✅ | ✅ |
| Motion brush | ✅ | ✅ | ✅ | - |
| Camera control | - | - | ✅ (simple) | - |

| Feature | v2-6 std | v2-6 pro |
|---------|----------|----------|
| Video generation | ✅ (no audio) | ✅ |
| Start/end frame | - | ✅ (no audio) |
| Voice control | - | ✅ |
| Motion control | ✅ | - |

---

## Gotchas

1. **Base64 format** - Do NOT include `data:image/png;base64,` prefix
2. **Motion brush limitations** - Only kling-v1, std/pro 5s mode
3. **Mask resolution** - Static and dynamic masks must have identical resolution
4. **End frame requires start** - Can't use `image_tail` without `image`
5. **Voice billing** - Tasks with voice_list billed at "with voice" rate
6. **cfg_scale** - Not supported for V2.x models
7. **Feature conflicts** - Start/end frames, motion brush, and camera control are mutually exclusive
