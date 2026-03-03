---
name: fal-model-search
description: Search and discover fal.ai model endpoints via their REST API. Use when needing to find available fal.ai models, get model metadata, retrieve OpenAPI schemas, or browse models by category. Triggers on "search fal models", "find fal endpoint", "list fal models", "fal model discovery", "what fal models are available".
---

# Fal Model Search

Search and discover fal.ai model endpoints using the `/v1/models` API.

## Quick Reference

**Base URL:** `https://api.fal.ai/v1/models`

**Authentication:** Optional (higher rate limits with API key)

**Modes:**
1. **List** - Paginated list of all models (no params)
2. **Find** - Get specific model(s) by endpoint_id
3. **Search** - Filter by query, category, or status

## Usage

### List All Models

```bash
curl "https://api.fal.ai/v1/models?limit=50"
```

### Find Specific Model

```bash
curl "https://api.fal.ai/v1/models?endpoint_id=fal-ai/flux/dev"
```

Multiple models:
```bash
curl "https://api.fal.ai/v1/models?endpoint_id=fal-ai/flux/dev&endpoint_id=fal-ai/flux-pro"
```

### Search by Category

```bash
curl "https://api.fal.ai/v1/models?category=text-to-image"
curl "https://api.fal.ai/v1/models?category=image-to-video"
curl "https://api.fal.ai/v1/models?category=text-to-video"
```

### Free-text Search

```bash
curl "https://api.fal.ai/v1/models?q=flux"
```

### Get OpenAPI Schema

Add `expand=openapi-3.0` to include full OpenAPI 3.0 specification:
```bash
curl "https://api.fal.ai/v1/models?endpoint_id=fal-ai/flux/dev&expand=openapi-3.0"
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | int | Max items to return (default varies by query type) |
| `cursor` | string | Pagination cursor from previous response |
| `endpoint_id` | string/array | Specific endpoint ID(s) to retrieve (1-50) |
| `q` | string | Free-text search query |
| `category` | string | Filter by category (text-to-image, image-to-video, etc.) |
| `status` | string | Filter by status: `active` or `deprecated` |
| `expand` | string | Expand fields: `openapi-3.0` for full schema |

## Response Structure

```json
{
  "models": [
    {
      "endpoint_id": "fal-ai/flux/dev",
      "metadata": {
        "display_name": "FLUX.1 [dev]",
        "category": "text-to-image",
        "description": "Fast text-to-image generation",
        "status": "active",
        "tags": ["fast", "pro"],
        "updated_at": "2025-01-15T12:00:00Z",
        "thumbnail_url": "https://fal.media/files/example.jpg",
        "model_url": "https://fal.run/fal-ai/flux/dev"
      },
      "openapi": { ... }  // Only when expand=openapi-3.0
    }
  ],
  "next_cursor": "Mg==",
  "has_more": true
}
```

## Common Categories

- `text-to-image` - Image generation from text
- `image-to-video` - Video from image + prompt
- `text-to-video` - Video from text prompt
- `image-to-3d` - 3D model from image
- `training` - Fine-tuning endpoints

## Example Endpoint IDs

- `fal-ai/flux/dev`
- `fal-ai/flux-pro`
- `fal-ai/wan/v2.2-a14b/text-to-video`
- `fal-ai/minimax/video-01/image-to-video`
- `fal-ai/hunyuan3d-v21`
- `fal-ai/bytedance/seedance/v1.5/pro/image-to-video`

## With Authentication

For higher rate limits, include API key:
```bash
curl -H "Authorization: Key YOUR_FAL_KEY" "https://api.fal.ai/v1/models"
```

## Documentation

Fetch complete docs index: https://docs.fal.ai/llms.txt
