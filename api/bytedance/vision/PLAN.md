# ByteDance Vision Models Plan

## Overview

Vision models use the **BytePlus Vision API** which is completely different from the ARK API used by seedream/seedance models.

---

## Key Differences from ARK API

| Aspect | ARK API (image/video) | Vision API |
|--------|----------------------|------------|
| Endpoint | `ark.ap-southeast.bytepluses.com` | `cv.byteplusapi.com` |
| Auth | Bearer token (`ARK_API_KEY`) | HMAC-SHA256 signature (Access Key + Secret Key) |
| SDK | `byteplussdkarkruntime.Ark` | `byteplus_sdk.visual.VisualService` |
| Submit | `client.content_generation.tasks.create()` | `visual_service.cv_submit_task()` |
| Poll | `client.content_generation.tasks.get()` | `visual_service.cv_get_result()` |
| Region | `ap-southeast` | `ap-singapore-1` |
| Service | `ark` | `cv` |

---

## Required Secrets

| Secret | Description |
|--------|-------------|
| `BYTEPLUS_ACCESS_KEY` | BytePlus Access Key (AK) |
| `BYTEPLUS_SECRET_KEY` | BytePlus Secret Key (SK) |

These are obtained from the BytePlus Console under IAM or API Key management.

---

## Models to Create

### 1. `omnihuman-1-0` (bytedance/omnihuman-1-0)

**Description**: Audio-driven avatar video generation. Takes a portrait image + audio and generates a video where the person speaks/sings in sync with the audio.

**req_key**: `realman_avatar_picture_omni_cv`

**Inputs**:
- `image` (required): Portrait image (JPG/PNG, <5MB, <4096x4096)
- `audio` (required): Audio file (<15 seconds recommended)

**Output**: MP4 video at 30 FPS

**Price**: $0.12/second of generated video

---

### 2. `omnihuman-1-5` (bytedance/omnihuman-1-5)

**Description**: Multi-character audio-driven video generation with subject detection. Can specify which character speaks using mask detection.

**req_keys**:
- Subject detection: `realman_avatar_object_detection_cv`
- Video generation: `realman_avatar_picture_omni15_cv`

**Inputs**:
- `image` (required): Portrait image with one or more characters
- `audio` (required): Audio file
- `mask_index` (optional): Which detected subject to drive (0 = largest, 1 = second largest, etc.)

**Output**: MP4 video at 30 FPS

**Workflow**:
1. Call subject detection to get mask URLs (sorted by area, largest first)
2. Select mask URL based on `mask_index`
3. Submit video generation task with `mask_url`
4. Poll for completion

---

## Directory Structure

```
./vision/
├── PLAN.md
├── vision_helper.py          # Shared helper for Vision API SDK
├── omnihuman-1-0-quick/
│   ├── inference.py
│   ├── inf.yml
│   └── requirements.txt
└── omnihuman-1-5/
    ├── inference.py
    ├── inf.yml
    └── requirements.txt
```

---

## Implementation Notes

### SDK Setup
```python
from byteplus_sdk.visual.VisualService import VisualService

visual_service = VisualService()
visual_service.set_ak(os.environ.get("BYTEPLUS_ACCESS_KEY"))
visual_service.set_sk(os.environ.get("BYTEPLUS_SECRET_KEY"))
```

### Subject Detection (OmniHuman-1.5 only)
```python
form = {
    "req_key": "realman_avatar_object_detection_cv",
    "image_url": "https://..."
}
resp = visual_service.cv_process(form)
# Parse resp["data"]["resp_data"] JSON to get mask URLs
```

### Video Generation
```python
# Submit task
form = {
    "req_key": "realman_avatar_picture_omni15_cv",
    "image_url": "https://...",
    "audio_url": "https://...",
    "mask_url": "https://..."  # optional, from subject detection
}
resp = visual_service.cv_submit_task(form)
task_id = resp["data"]["task_id"]

# Poll for result
form = {
    "req_key": "realman_avatar_picture_omni15_cv",
    "task_id": task_id
}
resp = visual_service.cv_get_result(form)
# Extract video URL from response
```

---

## Input Constraints

| Constraint | Value |
|------------|-------|
| Image format | JPG (preferred), PNG, JFIF |
| Image size | < 5 MB |
| Image resolution | < 4096x4096 |
| Audio duration | < 15 seconds recommended |
| Output format | MP4 |
| Output FPS | 30 |

---

## Status

- [ ] Create `vision_helper.py`
- [ ] Create `omnihuman-1-0-quick`
- [ ] Create `omnihuman-1-5`
- [ ] Deploy and test

---

## req_key Summary

| Model | req_key |
|-------|---------|
| OmniHuman 1.0 | `realman_avatar_picture_omni_cv` |
| OmniHuman 1.5 | `realman_avatar_picture_omni15_cv` |
| Subject Detection | `realman_avatar_object_detection_cv` |
