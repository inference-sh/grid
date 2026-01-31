# Kling AI - Video Effects API

Apply special effects to videos including dual-character interactions.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/effects` |
| Query Single | GET | `/v1/videos/effects/{task_id}` |
| Query List | GET | `/v1/videos/effects?pageNum=1&pageSize=30` |

---

## Available Effects

### Dual-Character Effects

| Effect | Description |
|--------|-------------|
| `hug` | Two characters hugging |
| `kiss` | Two characters kissing |
| `heart_gesture` | Two characters making heart gesture |

---

## Model Support

| Model | std 5s | std 10s | pro 5s | pro 10s |
|-------|--------|---------|--------|---------|
| kling-v1 | ✅ | ✅ | ✅ | ✅ |
| kling-v1-5 | ✅ | ✅ | ✅ | ✅ |
| kling-v1-6 | ✅ | ✅ | ✅ | ✅ |

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | No | `kling-v1` | Model version |
| `effect_name` | string | **Yes** | - | Effect type (`hug`, `kiss`, `heart_gesture`) |
| `images` | array | **Yes** | - | Two images for the effect |
| `mode` | string | No | `std` | `std` or `pro` |
| `duration` | string | No | `5` | `5` or `10` seconds |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

### images Array

```json
"images": [
  {"image_url": "https://..."},
  {"image_url": "https://..."}
]
```

Both images should contain clear faces for best results.

---

## Example

### Dual-Character Hug Effect

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/effects' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "effect_name": "hug",
    "images": [
        {"image_url": "https://example.com/person1.jpg"},
        {"image_url": "https://example.com/person2.jpg"}
    ],
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

## Gotchas

1. **Two images required** - Both images must contain clear faces
2. **Image quality matters** - High-quality, well-lit face photos work best
3. **Similar angle preferred** - Faces at similar angles produce better results
4. **Single face per image** - Each image should contain one clear face
