# BytePlus Apps Development Guide

This guide covers creating inference.sh apps that use BytePlus ARK SDK for video and image generation.

## Overview

BytePlus ModelArk provides access to powerful generative AI models like Seedance for video generation. Apps use the BytePlus Python SDK with an async polling pattern for content generation tasks.

## SDK Installation

```bash
pip install --upgrade 'byteplus-python-sdk-v2'
```

## Authentication

BytePlus apps require an API key from the BytePlus ModelArk console.

### inf.yml Configuration

```yaml
secrets:
  - key: ARK_API_KEY
    description: BytePlus ModelArk API key
    optional: false
```

### Accessing in Code

```python
import os
from byteplussdkarkruntime import Ark

client = Ark(
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)
```

## Using the Helper Module

copy the shared helper (byteplus_helper.py) into the app dir and import helper methods for common operations:

```python
from .byteplus_helper import (
    setup_byteplus_client,
    create_content_task,
    poll_task_status,
    download_video,
    build_text_content,
    build_image_content,
)
```

## Content Generation Pattern

BytePlus uses an async task-based pattern:

1. **Create Task**: Submit content generation request
2. **Poll Status**: Wait for task completion
3. **Download Result**: Get the generated content

### Example Flow

```python
async def run(self, input_data: AppInput) -> AppOutput:
    # 1. Build content list
    content = [
        build_text_content(
            input_data.prompt,
            duration=input_data.duration,
            camerafixed=input_data.camera_fixed
        ),
    ]

    # Add image if provided (for image-to-video)
    if input_data.image:
        content.append(build_image_content(input_data.image.uri))

    # 2. Create task
    task_id = create_content_task(
        self.client,
        model=self.model_id,
        content=content,
        logger=self.logger
    )

    # 3. Poll for completion
    result = await poll_task_status(
        self.client,
        task_id,
        logger=self.logger,
        cancel_flag_getter=lambda: self.cancel_flag
    )

    # 4. Download and return
    video_url = result.data.video.url  # Adjust based on actual response
    video_path = download_video(video_url, self.logger)

    return AppOutput(video=File(path=video_path))
```

## Cancellation Support

BytePlus tasks can be cancelled. Implement the `on_cancel` hook:

```python
class App(BaseApp):
    async def setup(self, config):
        self.cancel_flag = False
        self.current_task_id = None

    async def on_cancel(self):
        self.cancel_flag = True
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)
        return True
```

## Available Models

| Model ID | Type | Description |
|----------|------|-------------|
| `seedance-1-5-pro-251215` | Video | Seedance 1.5 Pro - High quality video generation |

## Content Format

### Text Prompt with Parameters

Parameters are embedded in the text using `--param value` syntax:

```python
content = [{
    "type": "text",
    "text": "A drone flying through mountains --duration 5 --camerafixed false"
}]
```

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--duration` | int | Video duration in seconds (typically 4-12) |
| `--camerafixed` | bool | Whether camera position is fixed |

### Image Content (First Frame)

```python
content = [
    {"type": "text", "text": "prompt --duration 5"},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/first_frame.png"}
    }
]
```

## Resource Configuration

BytePlus apps are API-based and don't require GPU:

```yaml
resources:
    gpu:
        count: 0
        vram: 0
        type: none
    ram: 4  # 4GB is typically sufficient
```

## Output Metadata

Track video generation for usage-based pricing:

```python
from inferencesh import OutputMeta, VideoMeta, VideoResolution

output_meta = OutputMeta(
    outputs=[
        VideoMeta(
            width=1280,
            height=720,
            resolution=VideoResolution.RES_720P,
            seconds=5.0,
            fps=24,
        )
    ]
)
```

## Error Handling

BytePlus tasks can fail. Check the error field:

```python
if result.status == "failed":
    error_msg = getattr(result, 'error', 'Unknown error')
    raise RuntimeError(f"Generation failed: {error_msg}")
```

## API Reference

- [BytePlus ModelArk Docs](https://docs.byteplus.com/en/docs/ModelArk)
- [Content Generation API](https://docs.byteplus.com/en/docs/ModelArk/1520757)
- [Task Query API](https://docs.byteplus.com/en/docs/ModelArk/1520757#query-content-generation-task-list-api)

## Example Apps

- `seedance-1-5-pro/` - Seedance video generation with image-to-video support
