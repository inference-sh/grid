# Pruna API Models

API base: `https://api.pruna.ai/v1`
Docs: `https://docs.api.pruna.ai/guides/models`

## Implemented Models (16)

### Image Generation

| Model | Docs | Price | Rate Limit |
|---|---|---|---|
| flux-dev | [docs](https://docs.api.pruna.ai/guides/models/flux-dev) | $0.005/img | 150/min |
| flux-dev-lora | [docs](https://docs.api.pruna.ai/guides/models/flux-dev-lora) | $0.01/img | 150/min |
| flux-klein-4b | [docs](https://docs.api.pruna.ai/guides/models/flux-2-klein-4b) | $0.0001/img | 150/min |
| p-image | [docs](https://docs.api.pruna.ai/guides/models/p-image) | $0.005/img | 500/min |
| p-image-lora | [docs](https://docs.api.pruna.ai/guides/models/p-image-lora) | - | 250/min |
| qwen-image | [docs](https://docs.api.pruna.ai/guides/models/qwen-image) | $0.025/img | 150/min |
| qwen-image-fast | [docs](https://docs.api.pruna.ai/guides/models/qwen-image-fast) | - | 150/min |
| wan-image-small | [docs](https://docs.api.pruna.ai/guides/models/wan-image-small) | $0.005/img | 150/min |
| z-image-turbo | [docs](https://docs.api.pruna.ai/guides/models/z-image-turbo) | - | 150/min |
| z-image-turbo-lora | [docs](https://docs.api.pruna.ai/guides/models/z-image-turbo-lora) | - | 150/min |

### Image Editing

| Model | Docs | Price | Rate Limit |
|---|---|---|---|
| p-image-edit | [docs](https://docs.api.pruna.ai/guides/models/p-image-edit) | $0.01/img | 500/min |
| p-image-edit-lora | [docs](https://docs.api.pruna.ai/guides/models/p-image-edit-lora) | - | 250/min |
| qwen-image-edit-plus | [docs](https://docs.api.pruna.ai/guides/models/qwen-image-edit-plus) | $0.03/img | 150/min |

### Video Generation

| Model | Docs | Price | Rate Limit |
|---|---|---|---|
| p-video | [docs](https://docs.api.pruna.ai/guides/models/p-video) | $0.005-0.04/sec | 250/min |
| wan-t2v | [docs](https://docs.api.pruna.ai/guides/models/wan-t2v) | $0.05/video (480p) | 30/min |
| wan-i2v | [docs](https://docs.api.pruna.ai/guides/models/wan-i2v) | $0.05/video (480p) | 30/min |

## Not Yet Implemented

| Model | Docs | Category |
|---|---|---|
| vace | [docs](https://docs.api.pruna.ai/guides/models/vace) | Video (character consistency) |
| p-image-trainer | [docs](https://docs.api.pruna.ai/guides/models/p-image-trainer) | LoRA training |
| p-image-edit-trainer | [docs](https://docs.api.pruna.ai/guides/models/p-image-edit-trainer) | LoRA training (editing) |

## Enum Parameters Reference

Parameters with fixed allowed values across Pruna models. Use this when adding new models or updating existing ones.

### aspect_ratio

Most models support a subset of these:

| Value | Used By |
|---|---|
| `1:1` | All models |
| `16:9` | All models |
| `9:16` | All models |
| `4:3` | All models |
| `3:4` | All models |
| `3:2` | Most models |
| `2:3` | Most models |
| `21:9` | flux-dev, flux-dev-lora, flux-klein-4b |
| `9:21` | flux-dev, flux-dev-lora, flux-klein-4b |
| `5:4` | flux-dev, flux-dev-lora, flux-klein-4b |
| `4:5` | flux-dev, flux-dev-lora, flux-klein-4b |
| `match_input_image` | p-image-edit, p-image-edit-lora, qwen-image-edit-plus, flux-klein-4b |
| `custom` | p-image, p-image-lora, qwen-image-fast, wan-image-small |

### output_format

`"png"`, `"jpg"`, `"webp"` -- most image models support all three.

### output_megapixels / megapixels

| Model | Allowed Values |
|---|---|
| flux-klein-4b (`output_megapixels`) | `"0.25"`, `"0.5"`, `"1"`, `"2"`, `"4"` |
| flux-dev-lora (`megapixels`) | `"1"`, `"0.25"` |

### resolution (video)

| Model | Allowed Values |
|---|---|
| p-video | `"720p"`, `"1080p"` |
| wan-t2v | `"480p"`, `"720p"` |
| wan-i2v | `"480p"`, `"720p"` |

### speed_mode

| Model | Allowed Values |
|---|---|
| flux-dev | `"Lightly Juiced (more consistent)"`, `"Juiced (default)"`, `"Extra Juiced (more speed)"`, `"Blink of an eye"` |
| flux-dev-lora | `"Base Model (compiled)"`, `"Lightly Juiced"`, `"Juiced"`, `"Extra Juiced"` |
| vace (not impl) | `"Lightly Juiced (more consistent)"`, `"Juiced (more speed)"`, `"Extra Juiced (even more speed)"` |

### image_size (qwen-image only)

`"optimize_for_quality"`, `"optimize_for_speed"`

### fps (p-video only)

`24`, `48`

### size (vace only, not implemented)

`"720*1280"`, `"1280*720"`, `"480*832"`, `"832*480"`

### sample_solver (vace only, not implemented)

`"unipc"`, `"dpm++"`

### training_type (p-image-trainer only, not implemented)

`"content"`, `"style"`, `"balanced"`
