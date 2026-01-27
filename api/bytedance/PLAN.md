# ByteDance Image Models & Video Updates Plan

## Overview

This plan covers:
1. Creating 4 new image generation models in `./image/`
2. Updating existing video models with input constraints

---

## Part 1: Image Models

### Key Differences from Video Models

| Aspect | Video | Image |
|--------|-------|-------|
| API | `client.content_generation.tasks.create()` | `client.images.generate()` |
| Pattern | Async polling | Synchronous response |
| Output | `result.content.video_url` | `result.data[0].url` |
| SDK | BytePlus ARK | BytePlus ARK (same) |

### Models to Create

#### 1. `seedream-4-5` (bytedance/seedream-4-5)
- **Model ID**: `seedream-4-5-251128`
- **Type**: Text-to-image + Image-to-image
- **Resolution**: 2560x1440 - 4096x4096 (2K-4K)
- **Features**: multi-image fusion, strong editing, clear text, 4K
- **Price**: $0.04/image
- **Parameters**:
  - `prompt` (required)
  - `image` (optional - enables I2I mode)
  - `size`: "2K" | "4K"
  - `sequential_image_generation`: "disabled" | "enabled"
  - `watermark`: bool
  - `response_format`: "url"

#### 2. `seedream-4-0` (bytedance/seedream-4-0)
- **Model ID**: `seedream-4-0-250828`
- **Type**: Text-to-image + Image-to-image
- **Resolution**: 1280x720 - 4096x4096
- **Features**: 4K ultra-HD, superior consistency
- **Price**: $0.03/image
- **Parameters**: Same as seedream-4-5

#### 3. `seededit-3-0-i2i` (bytedance/seededit-3-0-i2i)
- **Model ID**: `seededit-3-0-i2i-250628`
- **Type**: Image-to-image ONLY (editing)
- **Features**: Precise editing with content preservation
- **Price**: $0.03/image
- **Parameters**:
  - `prompt` (required) - editing instruction
  - `image` (required) - source image to edit
  - `size`: "adaptive" (preserves original)
  - `seed` (optional)
  - `guidance_scale`: float (e.g., 5.5)
  - `watermark`: bool

#### 4. `seedream-3-0-t2i` (bytedance/seedream-3-0-t2i)
- **Model ID**: `seedream-3-0-t2i-250415`
- **Type**: Text-to-image ONLY
- **Resolution**: 512x512 - 2048x2048
- **Features**: Cinematic quality, accurate text, native 2K
- **Price**: $0.03/image
- **Parameters**:
  - `prompt` (required)
  - `size`: "512x512" | "1024x1024" | "2048x2048" etc.
  - `response_format`: "url"

### Implementation Approach

1. **Create image helper** (`./byteplus_image_helper.py`):
   - `generate_image()` - calls `client.images.generate()`
   - No polling needed (synchronous)
   - Returns image URL directly

2. **Copy seedream-4-5 as template** (most feature-rich):
   - Supports both T2I and I2I modes
   - Then derive others from it

3. **Model-specific adjustments**:
   - `seededit-3-0-i2i`: image required, add guidance_scale/seed
   - `seedream-3-0-t2i`: no image support, different size options

### Directory Structure

```
./image/
├── seedream-4-5/
│   ├── inference.py
│   ├── byteplus_helper.py (copy from parent)
│   ├── inf.yml
│   └── requirements.txt
├── seedream-4-0/
│   └── ... (copy, adjust model ID and sizes)
├── seededit-3-0-i2i/
│   └── ... (image required, guidance_scale)
└── seedream-3-0-t2i/
    └── ... (text only, no image input)
```

---

## Part 2: Video Model Updates

### Current Limitations (from BytePlus docs)

| Model | Resolution | Duration | FPS |
|-------|------------|----------|-----|
| seedance-1-5-pro | 480P, 720P, 1080P | 4-12s | 24 |
| seedance-1-0-pro | 480P, 720P, 1080P | 2-12s | 24 |
| seedance-1-0-pro-fast | 480P, 720P, 1080P | 2-12s | 24 |
| seedance-1-0-lite | 480P, 720P, 1080P | 2-12s | 24 |

### Updates Needed

1. **Add ResolutionEnum** to all video models:
   ```python
   class ResolutionEnum(str, Enum):
       p480 = "480p"
       p720 = "720p"
       p1080 = "1080p"
   ```

2. **Update DurationEnum** per model:
   - `seedance-1-5-pro`: 4-12s (current: 4-10s) - add s11, s12
   - `seedance-1-0-pro`: 2-12s - add s2, s3, s11, s12
   - `seedance-1-0-pro-fast`: 2-12s - add s2, s3, s11, s12
   - `seedance-1-0-lite`: 2-12s - add s2, s3, s11, s12

3. **Add `resolution` input parameter** to each model

4. **Pass resolution to content** in `_build_content()`

---

## Execution Order

### Phase 1: Image Models
1. Create `./image/` directory
2. Create first model `seedream-4-5` as template:
   - Write `inference.py` with `client.images.generate()` pattern
   - Write `inf.yml` with category: image
   - Copy `byteplus_helper.py` and `requirements.txt`
3. Test pattern works
4. Copy and adjust for `seedream-4-0`
5. Create `seededit-3-0-i2i` (image required, guidance_scale)
6. Create `seedream-3-0-t2i` (text only)

### Phase 2: Video Model Updates
1. Update `seedance-1-5-pro`:
   - Add ResolutionEnum
   - Extend DurationEnum to 4-12s
   - Add resolution input
2. Update `seedance-1-0-pro`:
   - Add ResolutionEnum
   - Extend DurationEnum to 2-12s
3. Update `seedance-1-0-pro-fast`:
   - Same as 1-0-pro
4. Update `seedance-1-0-lite`:
   - Same as 1-0-pro

---

## Questions / Decisions

1. **Sequential image generation**: Default to "disabled"? (seems to be a batching feature)
2. **Size options**: Should we expose raw sizes or use enums like "2K", "4K"?
   - Recommendation: Use simple enums ("2K", "4K") for seedream-4-x
   - Use resolution strings ("1024x1024") for seedream-3-0-t2i
3. **Image output metadata**: Track dimensions? (for potential billing)
4. **Should we try BytePlus SDK for seededit first** before falling back to OpenAI SDK?
   - Yes, let's try BytePlus SDK - should work the same way

---

## Status

- [x] Phase 1: Image Models
  - [x] seedream-4-5 (T2I + I2I, 2K/4K sizes, model ID: seedream-4-5-251128)
  - [x] seedream-4-0 (T2I + I2I, 2K/4K sizes, model ID: seedream-4-0-250828)
  - [x] seedream-3-0-t2i (T2I only, 512-2048px sizes, model ID: seedream-3-0-t2i-250415)
  - [x] seededit-3-0-i2i (I2I editing only, guidance_scale, adaptive size, model ID: seededit-3-0-i2i-250628)
- [x] Phase 2: Video Model Updates
  - [x] seedance-1-5-pro (added ResolutionEnum 480p/720p/1080p, extended duration 4-12s)
  - [x] seedance-1-0-pro (added 480p, extended duration 2-12s)
  - [x] seedance-1-0-pro-fast (added 480p, extended duration 2-12s)
  - [x] seedance-1-0-lite (added ResolutionEnum, extended duration 2-12s, updated dimensions logic)
