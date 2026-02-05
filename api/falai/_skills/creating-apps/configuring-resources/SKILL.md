---
name: configuring-resources
description: Configure inf.yml for inference.sh apps. Use when setting GPU, VRAM, RAM, categories, environment variables, packages.txt, secrets, integrations, or resource requirements.
---

# Configuring Resources (inf.yml)

The `inf.yml` file defines app settings and resource requirements.

## Project Structure

```
my-app/
├── inf.yml           # Configuration
├── inference.py      # App logic
├── requirements.txt  # Python packages (pip)
└── packages.txt      # System packages (apt) — optional
```

## inf.yml Structure

```yaml
name: my-app
description: What my app does
category: image
kernel: python-3.11

resources:
  gpu:
    count: 1
    vram: 24    # 24GB (auto-converted)
    type: any
  ram: 32       # 32GB

env:
  MODEL_NAME: gpt-4
  HF_HUB_ENABLE_HF_TRANSFER: "1"
```

## Resource Units

CLI auto-converts human-friendly values:
- **< 1000** → GB (e.g., `80` = 80GB)
- **1000 to 1B** → MB

## GPU Types

`any` | `nvidia` | `amd` | `apple` | `none`

> **Note:** Currently only NVIDIA CUDA GPUs are supported.

## Categories

`image` | `video` | `audio` | `text` | `chat` | `3d` | `other`

## CPU-Only Apps

```yaml
resources:
  gpu:
    count: 0
    type: none
  ram: 4
```

## Secrets

Declare required secrets:

```yaml
secrets:
  - key: HF_TOKEN
    description: HuggingFace token for gated models
    optional: false
```

## Integrations

Declare OAuth integrations:

```yaml
integrations:
  - key: google.sheets
    description: Access to Google Sheets
    optional: true
```

## Dependencies

### Python Packages (requirements.txt)

```
torch>=2.0
transformers
accelerate
```

### System Packages (packages.txt)

For apt-installable dependencies:

```
ffmpeg
libgl1-mesa-glx
```

## Base Images

| Type | Image |
|------|-------|
| GPU | `docker.inference.sh/gpu:latest-cuda` |
| CPU | `docker.inference.sh/cpu:latest` |
