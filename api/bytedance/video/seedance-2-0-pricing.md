# Seedance 2.0 Series Pricing (BytePlus ModelArk)

## Token Formula (reverse-engineered, verified 2026-07-11)

```
tokens = out_w × out_h × (output_frames + input_video_frames) / 1024
```

- `output_frames = fps × seconds + 1` (BytePlus generates one extra frame: 5s@24fps = 121 frames)
- `input_video_frames = fps × input_seconds` (no +1 for input)
- Input video tokens use **output dimensions**, not input video's own resolution
- fps is always 24

### What affects token count

| Factor | Tokens? | Detail |
|--------|---------|--------|
| output resolution | yes | higher res = more tokens (w×h scales) |
| output duration | yes | more frames = more tokens |
| output aspect ratio | yes | changes w×h |
| reference video | yes | adds `input_fps × input_seconds` frames at output dimensions |
| images (any count, any role) | **no** | 0 tokens regardless of count |
| audio on/off | **no** | 0 token impact |
| +1 frame | **yes** | output always has fps×s+1 frames |

### What affects rate ($/M tokens)

Binary toggle only — any visual input (image, end_image, reference_images, reference_videos) triggers the lower "with video input" rate. Quantity and type don't matter beyond the toggle.

### Verified against BytePlus completion_tokens (12/12 exact match)

| test | res | ratio | s | input | tokens | predicted | match |
|------|-----|-------|---|-------|--------|-----------|-------|
| text-only | 480p | 16:9 | 5 | none | 50638 | 50638 | ✓ |
| first-frame img | 480p | 16:9 | 5 | 1 image | 50638 | 50638 | ✓ |
| 1 ref image | 480p | 16:9 | 5 | 1 ref img | 50638 | 50638 | ✓ |
| 2 ref images | 480p | 16:9 | 5 | 2 ref imgs | 50638 | 50638 | ✓ |
| first+last frame | 480p | 16:9 | 5 | 2 images | 50638 | 50638 | ✓ |
| ref video 5s | 480p | 16:9 | 5 | 5s video | 100858 | 100858 | ✓ |
| no audio | 480p | 16:9 | 5 | none | 50638 | 50638 | ✓ |
| 10s duration | 480p | 16:9 | 10 | none | 100858 | 100858 | ✓ |
| 1:1 ratio | 480p | 1:1 | 5 | none | 48400 | 48400 | ✓ |
| 720p text | 720p | 16:9 | 5 | none | 108900 | 108900 | ✓ |
| 720p + image | 720p | 16:9 | 5 | 1 image | 108900 | 108900 | ✓ |
| 720p + 480p vid | 720p | 16:9 | 5 | 5s 480p vid | 216900 | 216900 | ✓ |

## Per-Model Rates (USD/M tokens)

### Seedance 2.0

| Resolution | Text-only | With visual input |
|-----------|-----------|-------------------|
| 480p / 720p | $7.0 | $4.3 |
| 1080p | $7.7 | $4.7 |
| 4K | $4.0 | $2.4 |

### Seedance 2.0 Fast

| Resolution | Text-only | With visual input |
|-----------|-----------|-------------------|
| 480p / 720p | $5.6 | $3.3 |

### Seedance 2.0 Mini

| Resolution | Text-only | With visual input |
|-----------|-----------|-------------------|
| 480p / 720p | $3.5 | $2.1 |

## Resource Packs (Prepaid)

Resource packs are prepaid tokens used to deduct consumption from online inference. Purchase at [BytePlus Console](https://console.byteplus.com/common-buy/ModelArk%7C%7Cd7d6aanpgiftptb9ajcg).

> **Non-refundable.** If all packs expire or are depleted, excess consumption auto-converts to pay-as-you-go.

### Seedance 2.0

| Specification | Price (USD) | Min Purchase | Expires |
|---|---|---|---|
| 1M tokens | $4.30 | 7 packs minimum | 90 days |
| 10M tokens | $43.00 | No limit | 90 days |
| 100M tokens | $430.00 | No limit | 90 days |

### Seedance 2.0 Fast

| Specification | Price (USD) | Min Purchase | Expires |
|---|---|---|---|
| 1M tokens | $3.30 | 9 packs minimum | 90 days |
| 10M tokens | $33.00 | No limit | 90 days |
| 100M tokens | $330.00 | No limit | 90 days |

## CEL Pricing Notes

- `has()` guards required on `reference_images` and `reference_videos` before `size()` — CEL errors on missing fields cause entire expression to return 0 (text-only tasks fail silently)
- `video_seconds(inputs)` requires apps to populate `VideoMeta(seconds=dur)` with actual input video duration (via ffprobe)
- pricing agent needs dollar amounts, not microcents — let the agent handle internal conversion
- the +1 frame accounts for ~0.8% of output tokens at 5s; our CEL doesn't include it (acceptable margin)

## Key Notes

- Packs activate immediately on purchase (no manual activation needed).
- 90-day validity from purchase date; remaining tokens invalidate on expiry.
- Concurrent tasks: 10 per model.
- RPM: 600 requests/minute.
