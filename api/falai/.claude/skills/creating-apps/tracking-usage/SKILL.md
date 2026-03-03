---
name: tracking-usage
description: Track usage with output metadata in inference.sh apps. Use when implementing billing, counting tokens, or reporting image/video/audio generation metrics.
---

# Tracking Usage (Output Metadata)

Enable usage-based pricing by reporting what your app processes.

## Basic Structure

```python
from inferencesh import BaseAppOutput, OutputMeta, TextMeta, ImageMeta, VideoMeta, AudioMeta

class AppOutput(BaseAppOutput):
    result: File = Field(description="Generated output")
    # output_meta is inherited from BaseAppOutput
```

## MetaItem Types

| Type | Class | Fields |
|------|-------|--------|
| Text | `TextMeta` | `tokens` |
| Image | `ImageMeta` | `width`, `height`, `resolution_mp`, `steps`, `count` |
| Video | `VideoMeta` | `width`, `height`, `resolution`, `seconds` |
| Audio | `AudioMeta` | `seconds` |

## Examples

### LLM/Text Generation

Track both input (prompt) and output (completion) tokens:

```python
from inferencesh.models.usage import OutputMeta, TextMeta

return AppOutput(
    response=generated_text,
    output_meta=OutputMeta(
        inputs=[TextMeta(tokens=prompt_tokens)],
        outputs=[TextMeta(tokens=completion_tokens)]
    )
)
```

### Image Generation

```python
return AppOutput(
    image=File(path=output_path),
    output_meta=OutputMeta(
        outputs=[ImageMeta(
            width=1024,
            height=1024,
            resolution_mp=1.05,
            steps=20,
            count=1
        )]
    )
)
```

### Video Generation

```python
return AppOutput(
    video=File(path=output_path),
    output_meta=OutputMeta(
        outputs=[VideoMeta(
            width=1280,
            height=720,
            resolution="720p",
            seconds=5.0
        )]
    )
)
```

### Audio Generation

```python
return AppOutput(
    audio=File(path=output_path),
    output_meta=OutputMeta(
        outputs=[AudioMeta(seconds=30.0)]
    )
)
```

## Custom Data

Use `extra` for app-specific pricing factors:

```python
output_meta=OutputMeta(
    outputs=[ImageMeta(
        width=1024,
        height=1024,
        extra={
            "model": "sdxl-turbo",
            "lora_count": 2
        }
    )]
)
```

## Best Practices

1. **Always populate `output_meta`** if usage varies per request
2. **Use accurate token counts** from the actual tokenizer
3. **Report actual dimensions** - don't hardcode
4. **Include relevant `extra` data** for pricing flexibility
