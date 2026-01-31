# Kling AI - Elements API

Create and manage reusable elements for consistent character/object generation in Omni-Video.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Element | POST | `/v1/elements` |
| Query Element | GET | `/v1/elements/{element_id}` |
| List Elements | GET | `/v1/elements?pageNum=1&pageSize=30` |
| Delete Element | DELETE | `/v1/elements/{element_id}` |

---

## Description

Elements are pre-registered reusable objects (characters, items, styles) that can be referenced in Omni-Video prompts using `<<<element_1>>>`, `<<<element_2>>>` etc.

Benefits:
- Consistent character appearance across videos
- Reusable assets for multiple generations
- Better control over specific subjects

---

## Create Element

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Element name |
| `description` | string | No | Element description |
| `image_url` | string | **Yes** | Reference image URL |
| `type` | string | No | Element type (character, object, style) |

### Image Requirements

- Clear, high-quality image
- Single subject preferred
- Good lighting
- Formats: `.jpg`, `.jpeg`, `.png`
- Max size: 10MB

---

## Example

### Create Character Element

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/elements' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "name": "Main Character",
    "description": "Young woman with red hair",
    "image_url": "https://example.com/character.jpg",
    "type": "character"
}'
```

### Create Object Element

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/elements' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "name": "Magic Sword",
    "description": "Glowing blue sword with runes",
    "image_url": "https://example.com/sword.jpg",
    "type": "object"
}'
```

### List Elements

```bash
curl -X GET 'https://api-singapore.klingai.com/v1/elements?pageNum=1&pageSize=30' \
-H 'Authorization: Bearer xxx'
```

### Delete Element

```bash
curl -X DELETE 'https://api-singapore.klingai.com/v1/elements/12345' \
-H 'Authorization: Bearer xxx'
```

---

## Response

### Create Response

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "element_id": 12345,
    "name": "string",
    "description": "string",
    "type": "character",
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
    "elements": [
      {
        "element_id": 12345,
        "name": "string",
        "description": "string",
        "type": "character",
        "status": "ready",
        "created_at": 1722769557708
      }
    ],
    "total": 10,
    "page_num": 1,
    "page_size": 30
  }
}
```

---

## Using Elements in Omni-Video

### Single Element

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "<<<element_1>>> walking through a forest",
    "element_list": [
        {"element_id": 12345}
    ],
    "mode": "pro",
    "aspect_ratio": "16:9",
    "duration": "5"
}'
```

### Multiple Elements

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "<<<element_1>>> meets <<<element_2>>> in the city. <<<element_1>>> gives <<<element_3>>> to <<<element_2>>>",
    "element_list": [
        {"element_id": 12345},
        {"element_id": 67890},
        {"element_id": 11111}
    ],
    "mode": "pro",
    "aspect_ratio": "16:9",
    "duration": "7"
}'
```

### Combined with Images

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "<<<element_1>>> and <<<element_2>>> in the style of <<<image_1>>>",
    "element_list": [
        {"element_id": 12345},
        {"element_id": 67890}
    ],
    "image_list": [
        {"image_url": "https://example.com/style-ref.jpg"}
    ],
    "mode": "pro",
    "aspect_ratio": "1:1",
    "duration": "5"
}'
```

---

## Element Limits in Omni-Video

| Scenario | Max Elements + Images |
|----------|----------------------|
| With video reference | 4 total |
| Without video reference | 7 total |

---

## Gotchas

1. **Element ID is numeric** - Use integer, not string
2. **Processing time** - Elements need processing before use
3. **Status check** - Verify element status is "ready" before using
4. **Limit with video** - Fewer elements allowed when using video reference
5. **Numbering order** - `<<<element_1>>>` refers to first in element_list
6. **Image quality** - Better reference images = better consistency
