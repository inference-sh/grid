# Google Vertex AI Models

This directory contains inference.sh app implementations for Google Vertex AI models.

## Directory Structure

```
vertex/
├── DOCS.md                    # This documentation
├── vertex_helper.py           # Shared utilities (symlinked into each model dir)
├── gemini-2-5-flash-image/    # Gemini 2.5 Flash image model
├── gemini-3-pro-image-preview/ # Gemini 3 Pro image model
├── veo-3-1-fast/              # Veo 3.1 Fast video model
└── <model-name>/              # Future models
```

## Creating a New Model App

### 1. Initialize with infsh CLI

```bash
infsh app init <model-name>
cd <model-name>
```

This creates the directory structure with required boilerplate files.

### 2. Required Files

Each model directory needs:

| File | Description |
|------|-------------|
| `inference.py` | Main app logic (App class with setup/run methods) |
| `inf.yml` | App metadata (name, description, resources, integrations) |
| `input_schema.json` | Generated from AppInput pydantic model |
| `output_schema.json` | Generated from AppOutput pydantic model |
| `requirements.txt` | Python dependencies |
| `requirements.lock` | Locked dependency versions |
| `__init__.py` | Empty file for Python module |

### 3. Symlink Helper Module

Symlink `vertex_helper.py` into your model directory:

```bash
ln -s ../vertex_helper.py .
```

Then import what you need:

```python
from vertex_helper import (
    create_vertex_client,
    AspectRatioEnum,
    ResolutionEnum,
    OutputFormatEnum,
    SafetyToleranceEnum,
    calculate_dimensions,
    load_image_as_part,
    save_image_to_temp,
    build_image_generation_config,
    setup_logger,
)
```

### 4. inference.py Structure

```python
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List

# Import shared utilities
from vertex_helper import (
    create_vertex_client,
    AspectRatioEnum,
    # ... other imports
)

class AppInput(BaseAppInput):
    """Define input schema with pydantic Fields."""
    prompt: str = Field(description="...")
    # ... other fields

class AppOutput(BaseAppOutput):
    """Define output schema."""
    images: List[File] = Field(description="...")

class App(BaseApp):
    async def setup(self):
        """Initialize client and model. Called once on startup."""
        self.logger = setup_logger(__name__)
        self.metadata = metadata
        self.model_id = "your-model-id"
        self.client = create_vertex_client()

    async def run(self, input_data: AppInput) -> AppOutput:
        """Process a single request. Called for each inference."""
        # Your generation logic here
        pass
```

### 5. inf.yml Template

```yaml
namespace: google
name: model-name
description: Model description for Vertex AI
category: image  # or: video, audio, text
images:
    card: https://...
    thumbnail: ""
    banner: ""
metadata: {}
integrations:
  - key: gcp.vertex_ai
    description: Access Vertex AI models
resources:
    gpu:
        count: 0
        vram: 0
        type: none
    ram: 0
kernel: python-3.11
```

## vertex_helper.py Reference

### Enums

| Enum | Description |
|------|-------------|
| `OutputFormatEnum` | png, jpeg, webp |
| `OutputFormatExtendedEnum` | + heic, heif |
| `AspectRatioEnum` | Standard ratios (1:1, 16:9, etc.) |
| `AspectRatioAutoEnum` | + auto detection |
| `VideoAspectRatioEnum` | Video ratios (16:9, 9:16) |
| `SafetyToleranceEnum` | BLOCK_NONE to OFF |
| `ResolutionEnum` | 1K, 2K, 4K |
| `VideoResolutionEnum` | 720p, 1080p |

### Client Functions

| Function | Description |
|----------|-------------|
| `create_vertex_client(location, api_version)` | Create authenticated Vertex AI client |
| `setup_logger(name, level)` | Create configured logger |

### Image Utilities

| Function | Description |
|----------|-------------|
| `get_mime_type(file_path)` | Get MIME type from extension |
| `get_image_dimensions(file_path)` | Get (width, height) via PIL |
| `find_closest_aspect_ratio(w, h)` | Match dimensions to supported ratio |
| `calculate_dimensions(ratio, resolution)` | Get pixel dimensions |
| `load_image_as_part(file_path)` | Load image as Gemini Part |
| `load_video_as_part(file_path)` | Load video as Gemini Part |
| `save_image_to_temp(bytes, format)` | Save to temp file |

### Config Builders

| Function | Description |
|----------|-------------|
| `build_safety_settings(tolerance)` | Create SafetySetting list |
| `build_image_generation_config(...)` | Create GenerateContentConfig |
| `build_veo_payload(...)` | Create Veo video generation payload |

### Video Utilities (for Veo)

| Function | Description |
|----------|-------------|
| `detect_video_aspect_ratio(w, h)` | Detect 16:9 or 9:16 from dimensions |
| `resize_image_for_video(bytes, ratio)` | Resize image for video frame |
| `prepare_image_for_veo(path, ratio)` | Prepare image with base64 encoding |
| `save_video_to_temp(bytes, format)` | Save video to temp file |

### Long-Running Operations (Vertex AI REST API)

| Function | Description |
|----------|-------------|
| `get_vertex_credentials()` | Get (access_token, project) from env |
| `get_vertex_api_url(...)` | Build Vertex AI REST API URL |
| `start_long_running_operation(...)` | Start predictLongRunning call |
| `poll_long_running_operation(...)` | Poll until operation completes |
| `download_video_from_gcs(uri, token)` | Download video from GCS |

## Environment Variables

Required for all Vertex AI models:

| Variable | Description |
|----------|-------------|
| `GCP_ACCESS_TOKEN` | OAuth access token for authentication |
| `GCP_PROJECT_NUMBER` | GCP project number/ID |

These are automatically provided by the inference.sh platform via the `gcp.vertex_ai` integration.

## Model-Specific Notes

### Gemini Image Models (gemini-2-5-flash-image, gemini-3-pro-image-preview)

- Use `google.genai` SDK with `GenerateContentConfig`
- Support text-to-image and image-to-image editing
- Up to 14 input images for editing
- Output: 1-4 images per request
- Resolutions: 1K, 2K, 4K
- Various aspect ratios supported

### Veo Video Models (veo-3-1-fast)

- Use Vertex AI REST API (not google.genai SDK)
- Long-running operations: `predictLongRunning` + polling
- Support text-to-video and image-to-video
- Optional first frame and last frame for interpolation
- Aspect ratios: 16:9 (landscape) or 9:16 (portrait)
- Resolutions: 720p, 1080p
- Duration: 5-8 seconds
- Optional audio generation
- Output: 1-2 videos per request

## Generating Schemas

After defining `AppInput` and `AppOutput`, generate JSON schemas:

```bash
# From model directory
python -c "
from inference import AppInput, AppOutput
import json

with open('input_schema.json', 'w') as f:
    json.dump(AppInput.model_json_schema(), f, indent=2)

with open('output_schema.json', 'w') as f:
    json.dump(AppOutput.model_json_schema(), f, indent=2)
"
```

## Common Patterns

### Auto Aspect Ratio Detection

```python
from vertex_helper import find_closest_aspect_ratio, get_image_dimensions

if aspect_ratio == "auto" and input_images:
    w, h = get_image_dimensions(input_images[0].path)
    aspect_ratio = find_closest_aspect_ratio(w, h)
```

### Processing Gemini Response

```python
for part in response.candidates[0].content.parts:
    # Skip internal thought parts
    if hasattr(part, 'thought') and part.thought:
        continue

    if part.text is not None:
        # Handle text response
        descriptions.append(part.text)
    elif part.inline_data is not None:
        # Handle image/video output
        path = save_image_to_temp(part.inline_data.data, output_format)
        output_files.append(File(path=path))
```

### Error Handling

```python
async def run(self, input_data: AppInput) -> AppOutput:
    try:
        # ... generation logic
    except Exception as e:
        self.logger.error(f"Error: {e}")
        raise RuntimeError(f"Generation failed: {str(e)}")
```

### Video Generation with Veo (Long-Running Operation)

```python
from vertex_helper import (
    get_vertex_credentials,
    build_veo_payload,
    start_long_running_operation,
    poll_long_running_operation,
    download_video_from_gcs,
    save_video_to_temp,
)

# Get credentials
access_token, project = get_vertex_credentials()

# Build payload
payload = build_veo_payload(
    prompt="A cat playing piano",
    aspect_ratio="16:9",
    duration_seconds=8,
    resolution="720p",
    generate_audio=True,
)

# Start operation
response = await start_long_running_operation(
    access_token, project, "us-central1", "veo-3.1-fast-generate-001", payload
)
operation_name = response["name"]

# Poll until done
result = await poll_long_running_operation(
    access_token, project, "us-central1", "veo-3.1-fast-generate-001",
    operation_name, poll_interval=5.0, max_wait_time=600.0
)

# Process videos
for video in result["response"]["videos"]:
    if "gcsUri" in video:
        video_bytes = await download_video_from_gcs(video["gcsUri"], access_token)
        path = save_video_to_temp(video_bytes, "mp4")
```
