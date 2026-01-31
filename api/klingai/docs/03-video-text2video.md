# Kling AI - Text to Video API

Generate videos from text descriptions.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/text2video` |
| Query Single | GET | `/v1/videos/text2video/{task_id}` |
| Query List | GET | `/v1/videos/text2video?pageNum=1&pageSize=30` |

---

## Models

| Model | Description |
|-------|-------------|
| `kling-v1` | Base model (default) |
| `kling-v1-6` | V1.6 improved |
| `kling-v2-master` | V2 Master |
| `kling-v2-1-master` | V2.1 Master |
| `kling-v2-5-turbo` | V2.5 Turbo (faster) |
| `kling-v2-6` | V2.6 (supports sound) |

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | No | `kling-v1` | Model version |
| `prompt` | string | **Yes** | - | Text description (max 2500 chars) |
| `negative_prompt` | string | No | - | What to avoid (max 2500 chars) |
| `sound` | string | No | `off` | `on`/`off` - V2.6+ only |
| `cfg_scale` | float | No | `0.5` | Prompt adherence [0,1] - **not for V2.x** |
| `mode` | string | No | `std` | `std` (standard) or `pro` (professional) |
| `camera_control` | object | No | - | Camera movement settings |
| `aspect_ratio` | string | No | `16:9` | `16:9`, `9:16`, `1:1` |
| `duration` | string | No | `5` | `5` or `10` seconds |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID (must be unique) |

---

## Camera Control

### camera_control.type Options

| Type | Description | Needs config? |
|------|-------------|---------------|
| `simple` | Custom movement | Yes |
| `down_back` | Pan down and zoom out | No |
| `forward_up` | Zoom in and pan up | No |
| `right_turn_forward` | Rotate right and advance | No |
| `left_turn_forward` | Rotate left and advance | No |

### camera_control.config (for `simple` type)

**Choose ONE non-zero parameter:**

| Field | Range | Description |
|-------|-------|-------------|
| `horizontal` | [-10, 10] | Left (-) / Right (+) translation |
| `vertical` | [-10, 10] | Down (-) / Up (+) translation |
| `pan` | [-10, 10] | Rotation around X-axis |
| `tilt` | [-10, 10] | Rotation around Y-axis |
| `roll` | [-10, 10] | Rotation around Z-axis |
| `zoom` | [-10, 10] | Focal length change |

---

## Examples

### Basic Text to Video

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/text2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "prompt": "A majestic eagle soaring over snow-capped mountains at sunrise",
    "mode": "pro",
    "duration": "10",
    "aspect_ratio": "16:9"
}'
```

### With Negative Prompt

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/text2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "prompt": "Beautiful landscape with mountains and lake",
    "negative_prompt": "people, buildings, cars, text, watermark, blur",
    "mode": "pro",
    "duration": "5"
}'
```

### With Camera Control (Predefined)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/text2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "prompt": "Walking through a misty forest",
    "mode": "std",
    "duration": "5",
    "camera_control": {
        "type": "forward_up"
    }
}'
```

### With Camera Control (Custom)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/text2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "prompt": "Aerial view of a city at night",
    "mode": "std",
    "duration": "5",
    "camera_control": {
        "type": "simple",
        "config": {
            "horizontal": 0,
            "vertical": 0,
            "pan": 0,
            "tilt": 0,
            "roll": 0,
            "zoom": 5
        }
    }
}'
```

### With Sound (V2.6+)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/text2video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v2-6",
    "prompt": "Ocean waves crashing on rocky shore",
    "mode": "pro",
    "duration": "5",
    "sound": "on"
}'
```

---

## Response

### Create Task Response

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "task_id": "string",
    "task_status": "submitted",
    "task_info": {"external_task_id": "string"},
    "created_at": 1722769557708,
    "updated_at": 1722769557708
  }
}
```

### Query Task Response

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "task_id": "string",
    "task_status": "succeed",
    "task_status_msg": "string",
    "final_unit_deduction": "string",
    "task_info": {"external_task_id": "string"},
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

| Feature | v1 std 5s | v1 std 10s | v1 pro 5s | v1 pro 10s |
|---------|-----------|------------|-----------|------------|
| Video generation | ✅ | ✅ | ✅ | ✅ |
| Camera control | ✅ | - | - | - |

| Feature | v1-6 | v2-master | v2-1-master | v2-5-turbo | v2-6 |
|---------|------|-----------|-------------|------------|------|
| Video generation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sound | - | - | - | - | ✅ (pro only) |

---

## Gotchas

1. **cfg_scale not for V2.x** - Flexibility parameter doesn't work with V2 models
2. **Sound only V2.6+ pro** - The `sound` parameter only works with kling-v2-6 in pro mode
3. **Duration limits** - Only 5 or 10 seconds supported
4. **Prompt length** - Maximum 2500 characters
5. **Camera control support varies** - Only v1 std 5s supports camera control for T2V
6. **Choose one camera config** - For `simple` type, only one parameter should be non-zero
