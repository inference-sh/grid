# Pruna Implementation Plan

## Models to Implement

### Phase 1 - Core Models (Priority)
- [x] p-image - Text-to-image ($0.005/image, 500 req/min)
- [x] p-image-edit - Image editing ($0.010/image, 500 req/min)
- [x] p-video - Video generation ($0.005-0.04/sec based on resolution/draft)
- [x] p-image-lora - Image gen with LoRA ($0.005/image, 250 req/min)
- [x] p-image-edit-lora - Image edit with LoRA ($0.010/image, 250 req/min)

### Phase 2 - Trainers (Deferred)
- [ ] p-image-trainer - LoRA training ($1.80/1000 steps)
- [ ] p-image-edit-trainer - Edit LoRA training ($4.00/1000 steps)

Note: LoRAs must be trained with the matching trainer:
- p-image-lora requires LoRAs from p-image-trainer
- p-image-edit-lora requires LoRAs from p-image-edit-trainer

## API Details

### Base URL
```
https://api.pruna.ai/v1
```

### Authentication
Header: `apikey: {PRUNA_KEY}`

### Endpoints

1. **Create Prediction**
   ```
   POST /predictions
   Headers:
     - apikey: {key}
     - Model: {model-name}
     - Try-Sync: true (optional, 60s timeout)
     - Content-Type: application/json
   Body: { "input": { ... } }
   ```

2. **Upload File**
   ```
   POST /files
   Headers: apikey
   Body: multipart/form-data with content field
   Response: { "id": "...", "urls": { "get": "..." }, "metadata": { "width", "height" } }
   ```
   Files expire after 24 hours.

3. **Check Status**
   ```
   GET /predictions/status/{id}
   Headers: apikey
   Response: { "status": "starting|processing|succeeded|failed", "generation_url": "..." }
   ```

4. **Download Result**
   ```
   GET {generation_url}
   Headers: apikey
   Response: Binary file
   ```

## Model Parameters

### P-Image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description |
| `aspect_ratio` | string | "16:9" | 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, custom |
| `width` | int | - | 256-1440, multiple of 16 (only if aspect_ratio=custom) |
| `height` | int | - | 256-1440, multiple of 16 (only if aspect_ratio=custom) |
| `lora_weights` | string | - | HuggingFace URL (huggingface.co/owner/repo[/file.safetensors]) |
| `lora_scale` | number | 0.5 | -1 to 3 |
| `hf_api_token` | string | - | For private LoRAs |
| `prompt_upsampling` | bool | false | Enhance prompt with LLM |
| `seed` | int | random | For reproducibility |
| `disable_safety_checker` | bool | false | Disable safety filter |

### P-Image-LoRA

Same as P-Image, but `lora_weights` is effectively required.

### P-Image-Edit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Edit instruction |
| `images` | string[] | required | 1-5 image URLs |
| `turbo` | bool | true | Faster mode, set false for complex tasks |
| `aspect_ratio` | string | "match_input_image" | match_input_image, 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3 |
| `seed` | int | random | For reproducibility |
| `disable_safety_checker` | bool | false | Disable safety filter |

### P-Image-Edit-LoRA

Same as P-Image-Edit plus:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_weights` | string | required | HuggingFace URL |
| `lora_scale` | number | 1 | -1 to 3 |
| `hf_api_token` | string | - | For private LoRAs |

### P-Video

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description |
| `image` | string (URI) | - | Input image for i2v (ignores aspect_ratio) |
| `audio` | string (URI) | - | Audio file (ignores duration, flac/mp3/wav) |
| `duration` | int | 5 | 1-10 seconds (ignored if audio provided) |
| `resolution` | string | "720p" | 720p or 1080p |
| `fps` | int | 24 | 24 or 48 |
| `aspect_ratio` | string | "16:9" | 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 1:1 (ignored if image) |
| `draft` | bool | false | Lower quality preview (cheaper) |
| `save_audio` | bool | true | Include audio in output |
| `prompt_upsampling` | bool | true | Enhance prompt |
| `seed` | int | random | For reproducibility |
| `disable_safety_filter` | bool | true | Default is disabled |

## Implementation Notes

1. Use sync mode for fast image models, async polling for video
2. Secret key env var: `PRUNA_KEY`
3. Download results to temp files
4. Include OutputMeta for pricing:
   - ImageMeta for image models
   - VideoMeta with resolution/seconds for video (pricing varies by resolution+draft)
5. File uploads expire after 24 hours
6. LoRA weights must be HuggingFace URLs
