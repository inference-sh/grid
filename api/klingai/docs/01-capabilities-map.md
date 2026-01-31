# Kling AI - Capabilities Map

## Video Generation Models

### Omni-Video O1 (kling-video-o1)

| Feature | std | pro |
|---------|-----|-----|
| Video generation | ✅ | ✅ |

### Model Specifications

| Model | kling-v1 | kling-v1-5 | kling-v1-6 I2V | kling-v1-6 T2V | kling-v2-master |
|-------|----------|------------|----------------|----------------|-----------------|
| Mode | STD/PRO | STD/PRO | STD/PRO | STD/PRO | - |
| Resolution | 720p | 720p/1080p | 720p/1080p | 720p/1080p | 720p |
| Frame Rate | 30fps | 30fps | 30fps | 24fps | 24fps |

| Model | kling-v2-1 I2V | kling-v2-1-master | kling-v2-5 I2V | kling-v2-5 T2V |
|-------|----------------|-------------------|----------------|----------------|
| Mode | STD/PRO | - | PRO | PRO |
| Resolution | 720p/1080p | 1080p | 1080p | 1080p |
| Frame Rate | 24fps | 24fps | 24fps | 24fps |

---

## kling-v1 Features

| Feature | std 5s | std 10s | pro 5s | pro 10s |
|---------|--------|---------|--------|---------|
| **Text to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| camera control | ✅ | - | - | - |
| **Image to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| start/end frame | ✅ | - | ✅ | - |
| motion brush | ✅ | - | ✅ | - |
| **Video Extension** | ✅ | ✅ | ✅ | ✅ |
| **Video Effects** |
| Dual-character (Hug, Kiss, heart_gesture) | ✅ | ✅ | ✅ | ✅ |

> Note: Video extension does not support negative_prompt and cfg_scale

---

## kling-v1-5 Features

| Feature | std 5s | std 10s | pro 5s | pro 10s |
|---------|--------|---------|--------|---------|
| **Text to Video** | - | - | - | - |
| **Image to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| start/end frame | - | - | ✅ | ✅ |
| end frame | - | - | ✅ | ✅ |
| motion brush | - | - | ✅ | - |
| camera control (simple only) | - | - | ✅ | - |
| **Video Extension** | ✅ | ✅ | ✅ | ✅ |
| **Video Effects** |
| Dual-character | ✅ | ✅ | ✅ | ✅ |

---

## kling-v1-6 Features

| Feature | std 5s | std 10s | pro 5s | pro 10s |
|---------|--------|---------|--------|---------|
| **Text to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| **Image to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| start/end frame | - | - | ✅ | ✅ |
| end frame | - | - | ✅ | ✅ |
| **Multi-Image to Video** | ✅ | ✅ | ✅ | ✅ |
| **Multi-Elements** | ✅ | ✅ | ✅ | ✅ |
| **Video Extension** | ✅ | ✅ | ✅ | ✅ |
| **Video Effects** |
| Dual-character | ✅ | ✅ | ✅ | ✅ |

---

## kling-v2-master Features

| Feature | 5s | 10s |
|---------|-----|------|
| **Text to Video** | ✅ | ✅ |
| **Image to Video** | ✅ | ✅ |

---

## kling-v2-1 Features

| Feature | std 5s | std 10s | pro 5s | pro 10s |
|---------|--------|---------|--------|---------|
| **Text to Video** | - | - | - | - |
| **Image to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| start/end frame | - | - | ✅ | ✅ |

---

## kling-v2-1-master Features

| Feature | 5s | 10s |
|---------|-----|------|
| **Text to Video** | ✅ | ✅ |
| **Image to Video** | ✅ | ✅ |

---

## kling-v2-5-turbo Features

| Feature | std 5s | std 10s | pro 5s | pro 10s |
|---------|--------|---------|--------|---------|
| **Text to Video** | ✅ | ✅ | ✅ | ✅ |
| **Image to Video** |
| video generation | ✅ | ✅ | ✅ | ✅ |
| start/end frame | - | - | ✅ | ✅ |

---

## kling-v2-6 Features (with Audio)

| Feature | std 5s | std 10s | std other | pro 5s | pro 10s | pro other |
|---------|--------|---------|-----------|--------|---------|-----------|
| **Text to Video** |
| video generation | ✅ (no audio) | ✅ (no audio) | - | ✅ | ✅ | - |
| **Image to Video** |
| video generation | ✅ (no audio) | ✅ (no audio) | - | ✅ | ✅ | - |
| start/end frame | - | - | - | ✅ (no audio) | ✅ (no audio) | - |
| voice control | - | - | - | ✅ | ✅ | - |
| motion control | - | - | ✅ | - | - | ✅ |

---

## Model-Independent Features

| Feature | Support | Description |
|---------|---------|-------------|
| **Avatar** | ✅ | Generate digital human broadcast-style videos with one photo |
| **Lip Sync** | ✅ | Drive mouth shape with text or audio |
| **Video to Audio** | ✅ | Add audio to generated or uploaded videos |
| **Text to Audio** | - | Generate audio from text prompts |

---

## Image Generation Models

### kling-image-o1

| Aspect Ratio | 1:1 | 16:9 | 4:3 | 3:2 | 2:3 | 3:4 | 9:16 | 21:9 | auto |
|--------------|-----|------|-----|-----|-----|-----|------|------|------|
| text to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| image to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### kling-v1

| Aspect Ratio | 1:1 | 16:9 | 4:3 | 3:2 | 2:3 | 3:4 | 9:16 | 21:9 |
|--------------|-----|------|-----|-----|-----|-----|------|------|
| text to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| image to image (entire) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |

### kling-v1-5

| Aspect Ratio | 1:1 | 16:9 | 4:3 | 3:2 | 2:3 | 3:4 | 9:16 | 21:9 |
|--------------|-----|------|-----|-----|-----|-----|------|------|
| text to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| image to image (subject) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| image to image (face) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### kling-v2

| Aspect Ratio | 1:1 | 16:9 | 4:3 | 3:2 | 2:3 | 3:4 | 9:16 | 21:9 |
|--------------|-----|------|-----|-----|-----|-----|------|------|
| text to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| multi-image to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| restyle | ✅ (resolution = input) |

### kling-v2-new

| Feature | Support |
|---------|---------|
| text to image | - |
| restyle | ✅ (resolution = input) |

### kling-v2-1

| Aspect Ratio | 1:1 | 16:9 | 4:3 | 3:2 | 2:3 | 3:4 | 9:16 | 21:9 |
|--------------|-----|------|-----|-----|-----|-----|------|------|
| text to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| multi-image to image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Image Resolution

| Model | kling-v1 | kling-v1-5 | kling-v2 |
|-------|----------|------------|----------|
| Text to Image | 1K | 1K | 1K/2K |
| Image to Image | 1K | 1K | 1K |

---

## Additional Image Features

| Feature | Support | Description |
|---------|---------|-------------|
| **Image Expansion** | ✅ | Expand content based on existing images |
