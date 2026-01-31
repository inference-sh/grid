# Kling AI - Virtual Try-On API

Apply clothing items to person images for virtual fitting experiences.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/images/kolors-virtual-try-on` |
| Query Single | GET | `/v1/images/kolors-virtual-try-on/{task_id}` |
| Query List | GET | `/v1/images/kolors-virtual-try-on?pageNum=1&pageSize=30` |

---

## Description

Virtual Try-On allows users to see how clothing items would look on them. Use cases:
- E-commerce product visualization
- Fashion retail applications
- Personal styling
- Outfit planning

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_image` | string | **Yes** | - | Person image (URL or base64) |
| `cloth_image` | string | **Yes** | - | Clothing item image |
| `cloth_type` | string | No | `upper` | Clothing type |
| `n` | int | No | `1` | Number of results |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID |

### Clothing Types

| Type | Description |
|------|-------------|
| `upper` | Upper body (shirts, jackets) |
| `lower` | Lower body (pants, skirts) |
| `full` | Full body outfits |

---

## Image Requirements

### Model Image (Person)
- Full or half body visible
- Clear pose, not too obscured
- Good lighting
- Front-facing preferred
- Formats: `.jpg`, `.jpeg`, `.png`
- Max size: 10MB

### Cloth Image
- Clear product photo
- Minimal background preferred
- Good lighting
- Flat or mannequin display works best
- Same format requirements as model

---

## Examples

### Upper Body Try-On

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/kolors-virtual-try-on' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_image": "https://example.com/person.jpg",
    "cloth_image": "https://example.com/shirt.jpg",
    "cloth_type": "upper",
    "n": 1
}'
```

### Full Body Try-On

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/kolors-virtual-try-on' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_image": "https://example.com/person-fullbody.jpg",
    "cloth_image": "https://example.com/dress.jpg",
    "cloth_type": "full",
    "n": 2
}'
```

### Multiple Variations

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/images/kolors-virtual-try-on' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_image": "https://example.com/person.jpg",
    "cloth_image": "https://example.com/jacket.jpg",
    "cloth_type": "upper",
    "n": 4
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

## Concurrency

Virtual Try-On tasks consume **1 concurrency slot per task** (unlike image generation where it's based on `n`).

---

## Gotchas

1. **Pose matters** - Clear, unobstructed poses work best
2. **Cloth quality** - Clean product photos produce better results
3. **Matching type** - Use correct cloth_type for item category
4. **Body visibility** - Relevant body parts must be visible
5. **Background** - Simple backgrounds work better than complex ones
6. **Concurrency** - 1 slot per task regardless of `n` value
