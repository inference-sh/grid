# HappyHorse 1.0 Image-to-Video (happyhorse-1.0-i2v)

Generate physically realistic videos with smooth motion from a single image and optional text description using Alibaba's HappyHorse 1.0 I2V model via DashScope API.

## API Reference

- **Endpoint (Singapore):** `POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis`
- **Model:** `happyhorse-1.0-i2v`
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
| `prompt` | string | No | Text description, up to 5000 non-Chinese characters |
| `media` | array | Yes | Array with one `first_frame` image |

### Media Types

| Type | Description | Formats | Limits |
|------|-------------|---------|--------|
| `first_frame` | First frame image | JPEG, JPG, PNG, WEBP | >= 300px sides, 1:2.5 to 2.5:1 ratio, 10MB |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | string | `1080P` | `720P` or `1080P` |
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
    "model": "happyhorse-1.0-i2v",
    "input": {
        "prompt": "A cat running on the grass",
        "media": [
            {
                "type": "first_frame",
                "url": "https://cdn.translate.alibaba.com/r/wanx-demo-1.png"
            }
        ]
    },
    "parameters": {
        "resolution": "720P",
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

Input images are free. See [pricing.md](pricing.md) for full details.

## Notes

- Video generation takes 1-5 minutes
- `task_id` and `video_url` valid for 24 hours
- Video format: MP4 (H.264), 24fps
- Output aspect ratio follows the input image
- Duration range: 3-15 seconds
- Rate limit: 300 RPM
