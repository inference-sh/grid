# Kling AI - Omni-Video (O1) API

The Omni model (`kling-video-o1`) is a unified video generation system combining multiple capabilities through templated prompts.

## Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Create Task | POST | `/v1/videos/omni-video` |
| Query Single | GET | `/v1/videos/omni-video/{task_id}` |
| Query List | GET | `/v1/videos/omni-video?pageNum=1&pageSize=30` |

---

## Capabilities

1. **Image/Element Reference** - Generate with consistency using references
2. **Transformation** - Inpainting, outpainting, style changes, subject swaps
3. **Video Reference** - Use for camera movement, generate next/previous shots
4. **Start & End Frames** - Control first and last frames
5. **Text to Video** - Pure text-based generation

---

## Prompt Templates

Reference elements in prompts:
- `<<<element_1>>>`, `<<<element_2>>>` - Reference elements by ID
- `<<<image_1>>>`, `<<<image_2>>>` - Reference images by position
- `<<<video_1>>>` - Reference video

---

## Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | No | `kling-video-o1` | Must be O1 model |
| `prompt` | string | **Yes** | - | Templated prompt (max 2500 chars) |
| `image_list` | array | No | - | Reference images |
| `element_list` | array | No | - | Reference elements by ID |
| `video_list` | array | No | - | Reference videos |
| `mode` | string | No | `pro` | `std` or `pro` |
| `aspect_ratio` | string | Conditional | - | `16:9`, `9:16`, `1:1` |
| `duration` | string | No | `5` | `3`-`10` seconds |
| `watermark_info` | object | No | - | `{"enabled": true/false}` |
| `callback_url` | string | No | - | Webhook URL |
| `external_task_id` | string | No | - | Custom task ID (must be unique) |

### When is aspect_ratio required?
Required when **NOT** using:
- First-frame reference image
- Video editing (`refer_type: "base"`)

---

## image_list Structure

```json
"image_list": [
  {"image_url": "https://...", "type": "first_frame"},
  {"image_url": "https://...", "type": "end_frame"}
]
```

| Field | Description |
|-------|-------------|
| `image_url` | URL or base64 (no prefix) |
| `type` | `first_frame`, `end_frame`, or omit for general reference |

### Constraints
- Max 10MB per image, min 300px, ratio 1:2.5 to 2.5:1
- **With video reference**: max 4 images + elements combined
- **Without video**: max 7 images + elements combined
- End frame requires first frame
- Can't have >2 images with end frame

---

## element_list Structure

```json
"element_list": [
  {"element_id": 12345},
  {"element_id": 67890}
]
```

Elements are pre-registered objects (characters, items) by ID.

---

## video_list Structure

```json
"video_list": [
  {
    "video_url": "https://...",
    "refer_type": "feature",
    "keep_original_sound": "yes"
  }
]
```

| Field | Values | Description |
|-------|--------|-------------|
| `video_url` | URL | Video URL (required) |
| `refer_type` | `feature`, `base` | Feature=reference, Base=edit |
| `keep_original_sound` | `yes`, `no` | Keep original audio |

### Video Constraints
- Formats: `.mp4`, `.mov`
- Duration: 3-10 seconds
- Resolution: 720px-2160px
- Frame rate: 24-60 fps
- Max size: 200MB

---

## Duration Support

| Scenario | Duration Options |
|----------|------------------|
| Text-to-video | 5s, 10s |
| First/last frame | 5s, 10s |
| Video editing (`refer_type=base`) | Matches input video |
| Video + image/element reference | 3-10s |

---

## Examples

### Image/Element Reference

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "<<<image_1>>> strolling through Tokyo, encountered <<<element_1>>> and <<<element_2>>>. Style matches <<<image_2>>>",
    "image_list": [
        {"image_url": "https://..."},
        {"image_url": "https://..."}
    ],
    "element_list": [
        {"element_id": 12345},
        {"element_id": 67890}
    ],
    "mode": "pro",
    "aspect_ratio": "1:1",
    "duration": "7"
}'
```

### Video Transformation (Editing)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "Put the crown from <<<image_1>>> on the girl in <<<video_1>>>",
    "image_list": [{"image_url": "https://..."}],
    "video_list": [{
        "video_url": "https://...",
        "refer_type": "base",
        "keep_original_sound": "yes"
    }],
    "mode": "pro"
}'
```

### Video Reference (Style/Motion)

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "Referring to camera style in <<<video_1>>>, generate: <<<element_1>>> and <<<element_2>>> strolling in Tokyo",
    "image_list": [{"image_url": "https://..."}],
    "element_list": [
        {"element_id": 12345},
        {"element_id": 67890}
    ],
    "video_list": [{
        "video_url": "https://...",
        "refer_type": "feature"
    }],
    "mode": "pro",
    "aspect_ratio": "1:1",
    "duration": "7"
}'
```

### Extend Video (Next Shot)

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

### Start & End Frames

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "The person is dancing",
    "image_list": [
        {"image_url": "https://...", "type": "first_frame"},
        {"image_url": "https://...", "type": "end_frame"}
    ],
    "mode": "pro"
}'
```

### Pure Text to Video

```bash
curl -X POST 'https://api-singapore.klingai.com/v1/videos/omni-video' \
-H 'Authorization: Bearer xxx' \
-H 'Content-Type: application/json' \
-d '{
    "model_name": "kling-video-o1",
    "prompt": "A serene lake at sunset with birds flying",
    "mode": "pro",
    "aspect_ratio": "16:9",
    "duration": "7"
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
    "task_status": "submitted|processing|succeed|failed",
    "task_status_msg": "string",
    "task_info": {"external_task_id": "string"},
    "watermark_info": {"enabled": true},
    "final_unit_deduction": "string",
    "created_at": 1722769557708,
    "updated_at": 1722769557708,
    "task_result": {
      "videos": [{
        "id": "string",
        "url": "string",
        "watermark_url": "string",
        "duration": "string"
      }]
    }
  }
}
```

---

## Gotchas

1. **aspect_ratio required** when not using first-frame or video editing
2. **Video editing conflicts** - Can't use start/end frames with `refer_type=base`
3. **Reference limits** - With video: max 4, without: max 7 images+elements
4. **End frame needs first frame** - Can't specify only end frame
5. **Duration for editing** - Output matches input video duration
6. **Template order** - Elements numbered by order in lists
