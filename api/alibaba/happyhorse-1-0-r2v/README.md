# HappyHorse 1.0 Reference-to-Video (happyhorse-1.0-r2v)

Generate videos that preserve subject characters from up to 9 reference images, driven by a text prompt. Uses Alibaba's HappyHorse 1.0 R2V model via DashScope API.

## API Reference

- **Endpoint (Singapore):** `POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis`
- **Model:** `happyhorse-1.0-r2v`
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
| `prompt` | string | Yes | Text description with `[Image 1]`, `[Image 2]` references to media array order |
| `media` | array | Yes | Array of 1-9 `reference_image` entries |

### Media Types

| Type | Description | Formats | Limits |
|------|-------------|---------|--------|
| `reference_image` | Character/object reference | JPEG, JPG, PNG, WEBP | >= 400px shortest side, 10MB |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | string | `1080P` | `720P` or `1080P` |
| `ratio` | string | `16:9` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4` |
| `duration` | integer | 5 | Duration in seconds (3-15) |
| `watermark` | boolean | true | Add "HappyHorse" watermark |
| `seed` | integer | - | Random seed (0-2147483647) |

## Example

```bash
curl --location 'https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis' \
    -H 'X-DashScope-Async: enable' \
    -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "happyhorse-1.0-r2v",
    "input": {
        "prompt": "A woman in a red qipao [Image 1] gracefully raises her hand to unfold a folding fan [Image 2], while the tassel earrings [Image 3] sway lightly as she turns her head.",
        "media": [
            {
                "type": "reference_image",
                "url": "https://example.com/woman.jpg"
            },
            {
                "type": "reference_image",
                "url": "https://example.com/fan.jpg"
            },
            {
                "type": "reference_image",
                "url": "https://example.com/earrings.jpg"
            }
        ]
    },
    "parameters": {
        "resolution": "720P",
        "ratio": "16:9",
        "duration": 5
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

Input images are free (up to 9). See [pricing.md](pricing.md) for full details.

## Notes

- Video generation takes 1-5 minutes
- `task_id` and `video_url` valid for 24 hours
- Video format: MP4 (H.264)
- Reference images: 1-9, shortest side at least 400px, 720P+ recommended
- Use `[Image N]` in prompt to reference the Nth image in the media array
- You must specify the object from the reference image (e.g., "the woman in a red qipao in [Image 1]")
- Duration range: 3-15 seconds
- Rate limit: 300 RPM
