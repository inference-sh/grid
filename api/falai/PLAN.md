# FLUX LoRA Apps Implementation Plan

## Overview
Create TWO inference.sh apps:
1. **FLUX Dev LoRA** (`flux-dev-lora`) - FLUX.1 [dev] with LoRA support
2. **FLUX 2 Klein LoRA** (`flux-2-klein-lora`) - FLUX.2 [klein] 4B/9B with LoRA support

Both support text-to-image and image-to-image modes (auto-detected by image input).

---

# APP 1: FLUX Dev LoRA

## Endpoints
| Mode | Endpoint |
|------|----------|
| Text-to-Image | `fal-ai/flux-lora` |
| Image-to-Image | `fal-ai/flux-lora/image-to-image` |

## Parameters

| Parameter | Type | Default | Range/Options | Notes |
|-----------|------|---------|---------------|-------|
| `prompt` | string | - | - | **Required** |
| `image_size` | enum/object | `landscape_4_3` | Various presets or `{width, height}` | |
| `num_inference_steps` | int | 28 | 1-50 | |
| `seed` | int | random | - | Optional |
| `loras` | list | `[]` | LoraWeight objects | |
| `guidance_scale` | float | 3.5 | 0-35 | |
| `num_images` | int | 1 | 1-4 | |
| `enable_safety_checker` | bool | true | | |
| `output_format` | enum | `jpeg` | `jpeg`, `png` | |

### Image-to-Image Additional Parameters
| Parameter | Type | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| `image_url` | string | - | - | **Required for I2I** (single URL) |
| `strength` | float | 0.85 | 0.01-1.0 | 1.0 = full remake, 0.0 = preserve original |

### LoraWeight Object
```json
{
  "path": "https://url-to-lora.safetensors",
  "scale": 1.0
}
```

---

# APP 2: FLUX 2 Klein LoRA

## Endpoints
| Mode | Model Size | Endpoint |
|------|------------|----------|
| Text-to-Image | 4B | `fal-ai/flux-2/klein/4b/base/lora` |
| Edit (I2I) | 4B | `fal-ai/flux-2/klein/4b/base/edit/lora` |
| Text-to-Image | 9B | `fal-ai/flux-2/klein/9b/base/lora` |
| Edit (I2I) | 9B | `fal-ai/flux-2/klein/9b/base/edit/lora` |

## Parameters

| Parameter | Type | Default | Range/Options | Notes |
|-----------|------|---------|---------------|-------|
| `prompt` | string | - | - | **Required** |
| `negative_prompt` | string | `""` | - | What to avoid |
| `model_size` | enum | `4b` | `4b`, `9b` | |
| `image_size` | enum/object | `landscape_4_3` | Various presets or `{width, height}` | |
| `num_inference_steps` | int | 28 | 4-50 | |
| `seed` | int | random | - | Optional |
| `loras` | list | `[]` | Max 3 LoRAs | |
| `guidance_scale` | float | 5.0 | 0-20 | |
| `num_images` | int | 1 | 1-4 | |
| `acceleration` | enum | `regular` | `none`, `regular`, `high` | |
| `enable_safety_checker` | bool | true | | |
| `output_format` | enum | `png` | `jpeg`, `png`, `webp` | |

### Image-to-Image Additional Parameters
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `image_urls` | list[string] | - | **Required for I2I** (max 4 URLs) |

### LoRAInput Object (Klein)
```json
{
  "path": "https://url-to-lora.safetensors",
  "scale": 1.0
}
```

---

# Key Differences Between Apps

| Feature | FLUX Dev LoRA | FLUX 2 Klein LoRA |
|---------|---------------|-------------------|
| Model sizes | Single | 4B, 9B |
| I2I input | `image_url` (single) | `image_urls` (list, max 4) |
| I2I strength | `strength` param (0.01-1.0) | None |
| Negative prompt | No | Yes |
| Acceleration | No | Yes (`none`/`regular`/`high`) |
| Guidance range | 0-35 (default 3.5) | 0-20 (default 5.0) |
| Output formats | jpeg, png | jpeg, png, webp |

---

# Implementation Steps

1. **For each app:**
   ```bash
   cd /home/ok/inference/grid/api/falai
   infsh app init <app-name>
   cp fal_helper.py <app-name>/
   ```

2. **Create files:**
   - `inf.yml` - config
   - `requirements.txt` - dependencies
   - `inference.py` - main app logic
   - `__init__.py` - empty

3. **Logic flow in inference.py:**
   ```
   if image input provided:
       use I2I endpoint
   else:
       use T2I endpoint
   ```

---

# File Templates

## inf.yml (both apps)
```yaml
namespace: falai
name: <app-name>
description: "<description>"
category: image
kernel: python-3.11
resources:
    gpu:
        count: 0
        vram: 0
        type: none
    ram: 4000000000
secrets:
  - key: FAL_KEY
    optional: false
```

## requirements.txt (both apps)
```txt
pydantic >= 2.0.0
inferencesh
fal-client>=0.4.0
requests>=2.28.0
```

---

# Current Status

- [x] `flux-2-klein-lora` - Complete
- [x] `flux-dev-lora` - Complete

Both apps have been implemented with proper `infsh app init` scaffolding.
