---
name: tracking-usage
description: Track usage with output metadata in inference.sh apps. Use when implementing billing, counting tokens, or reporting image/video/audio generation metrics.
---

# Tracking Usage (Output Metadata)

Enable usage-based pricing by reporting what your app processes.

## MetaItem Types

| Type | Class | Fields |
|------|-------|--------|
| Text | `TextMeta` | `tokens` |
| Image | `ImageMeta` | `width`, `height`, `steps`, `count` |
| Video | `VideoMeta` | `width`, `height`, `seconds` |
| Audio | `AudioMeta` | `seconds` |

## Examples

```python
from inferencesh.models.usage import OutputMeta, TextMeta, ImageMeta, VideoMeta, AudioMeta

# LLM tokens
output_meta=OutputMeta(
    inputs=[TextMeta(tokens=prompt_tokens)],
    outputs=[TextMeta(tokens=completion_tokens)]
)

# Image generation
output_meta=OutputMeta(
    outputs=[ImageMeta(width=1024, height=1024, steps=20, count=1)]
)

# Video generation
output_meta=OutputMeta(
    outputs=[VideoMeta(width=1280, height=720, seconds=5.0)]
)

# Audio generation
output_meta=OutputMeta(
    outputs=[AudioMeta(seconds=30.0)]
)
```

## Custom Data

```python
output_meta=OutputMeta(
    outputs=[ImageMeta(
        width=1024, height=1024,
        extra={"model": "sdxl-turbo", "lora_count": 2}
    )]
)
```

📖 **Full docs**: [inference.sh/docs/extend/output-meta](https://inference.sh/docs/extend/output-meta)
