# Kling AI - Video Extension API

Extend existing videos by generating additional content.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/video-extend` |
| Query Single | GET | `/v1/videos/video-extend/{task_id}` |
| Query List | GET | `/v1/videos/video-extend?pageNum=1&pageSize=30` |

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
| `video_id` | string | **Yes** | - | ID of source video to extend |
| `prompt` | string | No | - | Description for extended content |
| `mode` | string | No | `std` | `std` or `pro` |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

---

## Limitations

**Not supported for video extension:**
- `negative_prompt`
- `cfg_scale`

---

## Example

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/video-extend' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "video_id": "original-video-id",
    "prompt": "The scene continues with more action",
    "mode": "pro"
}'
```

---

## Alternative: Omni-Video Extension

You can also extend videos using the Omni-Video API with feature reference:

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "Based on <<<video_1>>>, generate the next shot",
    "video_list": [{
        "video_url": "https://...",
        "refer_type": "feature"
    }],
    "mode": "pro"
}'
```

This method provides more control over the extended content.

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

1. **Source video required** - Must reference an existing Kling video by ID
2. **No negative_prompt** - Not supported for extension
3. **No cfg_scale** - Not supported for extension
4. **Omni alternative** - Omni-Video provides more flexibility for extension
