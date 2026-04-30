# HappyHorse 1.0 Text-to-Video (happyhorse-1.0-t2v)

Generate physically realistic videos with smooth motion from text prompts using Alibaba's HappyHorse 1.0 T2V model via DashScope API.

## API Reference

- **Endpoint (Singapore):** `POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis`
- **Model:** `happyhorse-1.0-t2v`
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
| `prompt` | string | Yes | Text description, up to 5000 non-Chinese characters |

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
    "model": "happyhorse-1.0-t2v",
    "input": {
        "prompt": "A miniature city built from cardboard and bottle caps comes alive at night. A cardboard train rolls slowly by, dotted with tiny lights that illuminate the path ahead."
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

See [pricing.md](pricing.md) for full details.

## Notes

- Video generation takes 1-5 minutes
- `task_id` and `video_url` valid for 24 hours
- Video format: MP4 (H.264), 24fps
- Duration range: 3-15 seconds
- Rate limit: 300 RPM
