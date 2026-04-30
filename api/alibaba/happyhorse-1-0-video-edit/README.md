# HappyHorse 1.0 Video Edit (happyhorse-1.0-video-edit)

Edit videos through natural language instructions with up to 5 reference images. Supports local or global editing while preserving original motion dynamics. Uses Alibaba's HappyHorse 1.0 Video Edit model via DashScope API.

## API Reference

- **Endpoint (Singapore):** `POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis`
- **Model:** `happyhorse-1.0-video-edit`
- **Protocol:** Asynchronous (create task -> poll for result)

## Request Parameters

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `Authorization` | Yes | `Bearer $DASHSCOPE_API_KEY` |
| `X-DashScope-Async` | Yes | Must be `enable` |

### Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Editing instruction, up to 5000 non-Chinese characters |
| `media` | array | Yes | Exactly 1 `video` + 0-5 `reference_image` entries |

### Media Types

| Type | Description | Formats | Limits |
|------|-------------|---------|--------|
| `video` | Video to edit | MP4, MOV (H.264) | 3-60s, longer side <= 2160px, shorter >= 320px, 100MB, > 8fps |
| `reference_image` | Reference for edits | JPEG, JPG, PNG, WEBP | >= 300px sides, 1:2.5 to 2.5:1, 10MB |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | string | `1080P` | `720P` or `1080P` |
| `watermark` | boolean | true | Add "HappyHorse" watermark |
| `audio_setting` | string | `auto` | `auto` (model decides) or `origin` (keep original) |
| `seed` | integer | - | Random seed (0-2147483647) |

## Example

```bash
curl --location 'https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis' \
    -H 'X-DashScope-Async: enable' \
    -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "happyhorse-1.0-video-edit",
    "input": {
        "prompt": "Make the character wear the striped sweater from the image",
        "media": [
            {
                "type": "video",
                "url": "https://example.com/input.mp4"
            },
            {
                "type": "reference_image",
                "url": "https://example.com/sweater.jpg"
            }
        ]
    },
    "parameters": {
        "resolution": "720P"
    }
}'
```

## Polling

```bash
curl -X GET https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id} \
    --header "Authorization: Bearer $DASHSCOPE_API_KEY"
```

Poll every 15 seconds. Status: PENDING -> RUNNING -> SUCCEEDED/FAILED.

## Pricing

| Resolution | Price per second |
|------------|-----------------|
| 720P | $0.14 |
| 1080P | $0.24 |

Both input and output video seconds are billed. See [pricing.md](pricing.md) for full details.

## Notes

- Video editing takes 1-5 minutes
- `task_id` and `video_url` valid for 24 hours
- Video format: MP4 (H.264)
- Input video: 3-60 seconds, but output capped at 15 seconds (if input > 15s, only first 15s used)
- Reference images: 0-5
- Both input and output video duration are billed
- Rate limit: 300 RPM
