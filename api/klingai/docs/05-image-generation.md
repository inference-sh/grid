# Kling AI - Image Generation API

Generate images from text descriptions with optional reference images.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/images/generations` |
| Query Single | GET | `/v1/images/generations/{task_id}` |
| Query List | GET | `/v1/images/generations?pageNum=1&pageSize=30` |

---

## Models

| Model | Description |
|-------|-------------|
| `kling-v1` | Base model (default) |
| `kling-v1-5` | V1.5 (supports image reference) |
| `kling-v2` | V2 (multi-image, restyle) |
| `kling-v2-new` | V2 New (restyle only) |
| `kling-v2-1` | V2.1 |

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | No | `kling-v1` | Model version |
| `prompt` | string | **Yes** | - | Text description (max 2500 chars) |
| `negative_prompt` | string | No | - | What to avoid (not with image ref) |
| `image` | string | No | - | Reference image (URL or base64) |
| `image_reference` | string | No | - | `subject` or `face` |
| `image_fidelity` | float | No | `0.5` | Reference strength [0,1] |
| `human_fidelity` | float | No | `0.45` | Face similarity [0,1] (subject only) |
| `resolution` | string | No | `1k` | `1k` or `2k` |
| `n` | int | No | `1` | Number of images [1,9] |
| `aspect_ratio` | string | No | `16:9` | Output ratio |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

---

## Aspect Ratios

| Ratio | Models |
|-------|--------|
| `16:9` | All |
| `9:16` | All |
| `1:1` | All |
| `4:3` | All |
| `3:4` | All |
| `3:2` | All |
| `2:3` | All |
| `21:9` | v1-5, v2, v2-1 |

---

## Image Reference (V1.5 only)

### Reference Types

| Type | Description |
|------|-------------|
| `subject` | Character feature reference |
| `face` | Face appearance reference (single face required) |

### Reference Parameters

| Field | Range | Description |
|-------|-------|-------------|
| `image_fidelity` | [0,1] | Overall reference strength |
| `human_fidelity` | [0,1] | Facial similarity (subject only) |

### Requirements
- `image_reference` requires `image` parameter
- Only `kling-v1-5` supports image reference
- Face reference requires exactly one face in image
- **Negative prompt not supported with image reference**

---

## Resolution

| Model | Text to Image | Image to Image |
|-------|---------------|----------------|
| kling-v1 | 1K | 1K |
| kling-v1-5 | 1K | 1K |
| kling-v2 | 1K/2K | 1K |

---

## Examples

### Basic Text to Image

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/generations' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "prompt": "A futuristic cityscape at night with neon lights and flying cars",
    "n": 4,
    "aspect_ratio": "16:9",
    "resolution": "1k"
}'
```

### With Negative Prompt

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/generations' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1",
    "prompt": "Beautiful landscape with mountains and lake",
    "negative_prompt": "people, buildings, cars, text, watermark, blur, low quality",
    "n": 2,
    "aspect_ratio": "16:9"
}'
```

### Character Reference (Subject)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/generations' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1-5",
    "prompt": "The person in a superhero costume flying through the sky",
    "image": "https://example.com/reference.jpg",
    "image_reference": "subject",
    "image_fidelity": 0.7,
    "human_fidelity": 0.5,
    "n": 2,
    "aspect_ratio": "16:9"
}'
```

### Face Reference

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/generations' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v1-5",
    "prompt": "Portrait in Renaissance painting style",
    "image": "https://example.com/face.jpg",
    "image_reference": "face",
    "image_fidelity": 0.8,
    "n": 1,
    "aspect_ratio": "3:4"
}'
```

### High Resolution (V2)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/generations' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-v2",
    "prompt": "Detailed macro photography of a butterfly wing",
    "n": 1,
    "aspect_ratio": "1:1",
    "resolution": "2k"
}'
```

---

## Response

### Create Task

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

### Query Task

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
    "created_at": 1722769557708,
    "updated_at": 1722769557708,
    "task_result": {
      "images": [
        {"index": 0, "url": "https://..."},
        {"index": 1, "url": "https://..."}
      ]
    },
    "task_info": {"external_task_id": "string"}
  }
}
```

---

## Model Feature Support

### kling-v1

| Feature | Support |
|---------|---------|
| Text to image | ✅ |
| Image to image (entire) | ✅ |
| 21:9 ratio | ❌ |

### kling-v1-5

| Feature | Support |
|---------|---------|
| Text to image | ✅ |
| Image to image (subject) | ✅ |
| Image to image (face) | ✅ |
| 21:9 ratio | ✅ |

### kling-v2

| Feature | Support |
|---------|---------|
| Text to image | ✅ |
| Multi-image to image | ✅ |
| Restyle | ✅ (output = input resolution) |
| 2K resolution | ✅ |

### kling-v2-new

| Feature | Support |
|---------|---------|
| Text to image | ❌ |
| Restyle | ✅ (output = input resolution) |

### kling-v2-1

| Feature | Support |
|---------|---------|
| Text to image | ✅ |
| Multi-image to image | ✅ |

---

## Concurrency Note

Image generation concurrency = `n` parameter value.

If `n=9`, the task occupies **9 concurrency slots**.

---

## Gotchas

1. **No negative prompt with image reference** - Can't use both together
2. **Face reference single face** - Image must contain exactly one face
3. **V1.5 for image reference** - Only kling-v1-5 supports image_reference
4. **Concurrency consumption** - Based on `n` value, not task count
5. **Resolution support varies** - Only V2 supports 2K
6. **Restyle resolution** - Output matches input resolution
7. **Content cleared after 30 days** - Save generated images promptly
