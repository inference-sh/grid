# Kling AI - Image Expansion API

Expand images by generating content beyond their original boundaries (outpainting).

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/images/expand` |
| Query Single | GET | `/v1/images/expand/{task_id}` |
| Query List | GET | `/v1/images/expand?pageNum=1&pageSize=30` |

---

## Description

Image Expansion (outpainting) extends images by generating new content that seamlessly continues the original image. Use cases:
- Extending cropped photos
- Creating wider panoramas
- Adding context to images
- Adjusting aspect ratios

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | string | **Yes** | - | Source image (URL or base64) |
| `prompt` | string | No | - | Description for expanded area |
| `direction` | string | **Yes** | - | Expansion direction(s) |
| `expansion_ratio` | float | No | `1.5` | How much to expand |
| `n` | int | No | `1` | Number of variations |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

### Direction Options

| Direction | Description |
|-----------|-------------|
| `left` | Expand left side |
| `right` | Expand right side |
| `top` | Expand top |
| `bottom` | Expand bottom |
| `all` | Expand all directions |

Multiple directions can be combined: `left,right` or `top,bottom`

---

## Examples

### Expand Right

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/expand' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "image": "https://example.com/landscape.jpg",
    "direction": "right",
    "expansion_ratio": 1.5,
    "prompt": "Continue the mountain landscape with more trees"
}'
```

### Expand All Directions

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/expand' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "image": "https://example.com/portrait.jpg",
    "direction": "all",
    "expansion_ratio": 1.3,
    "prompt": "Office environment with desk and window"
}'
```

### Expand Top and Bottom

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/expand' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "image": "https://example.com/cropped.jpg",
    "direction": "top,bottom",
    "expansion_ratio": 1.4,
    "n": 3
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
      "images": [
        {"index": 0, "url": "https://..."},
        {"index": 1, "url": "https://..."}
      ]
    }
  }
}
```

---

## Gotchas

1. **Prompt helps** - Describing desired content improves results
2. **Edge matching** - AI matches existing edges for seamless expansion
3. **Ratio limits** - Very large ratios may reduce quality
4. **Original preserved** - Original image content remains unchanged
5. **Multiple variations** - Use `n` parameter to get different options
