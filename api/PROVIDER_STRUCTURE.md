# Grid API Provider Structure Guide

Reference document for adding new API providers to `grid/api/`.

## Directory Structure

```
grid/api/
├── {provider}/                 # Provider folder (e.g., pruna, falai, alibaba)
│   ├── {provider}_helper.py    # Shared helper module for all models
│   ├── .claude/                # Optional: Claude instructions
│   ├── {model-name}/           # One folder per model/app
│   │   ├── inference.py        # Main app logic
│   │   ├── inf.yml             # App configuration (namespace, secrets, etc.)
│   │   ├── requirements.txt    # Python dependencies
│   │   ├── input_schema.json   # Auto-generated from AppInput
│   │   ├── output_schema.json  # Auto-generated from AppOutput
│   │   ├── __init__.py         # Package init
│   │   ├── README.md           # Model documentation
│   │   ├── pricing.md          # Optional: Pricing notes
│   │   └── {provider}_helper.py -> ../{provider}_helper.py  # Symlink to shared helper
│   └── ...
```

## Creating New Apps

```bash
cd grid/api/{provider}
infsh app init {model-name}
```

This creates a new folder with default template files.

## Key Files

### inf.yml

```yaml
namespace: {provider}           # Provider name (e.g., pruna)
name: {model-name}              # Model/app name (e.g., p-image)
description: ...                # Human-readable description
metadata: {}
category: image|video|text|...  # App category
images:
    card: https://...           # Card thumbnail
    thumbnail: ""
    banner: ""
env: {}
kernel: python-3.11             # Or empty for API-only apps
secrets:
    - key: {PROVIDER}_KEY       # Secret env var name (e.g., PRUNA_KEY)
      description: ...
resources:
    gpu:
        count: 0                # 0 for API-only apps
        vram: 0
        type: none
    ram: 0
```

### Helper Module Pattern

Shared helpers are symlinked into each model folder:

```bash
# In model folder
ln -s ../{provider}_helper.py .
```

Helper modules typically include:
- API client setup (reading secret from env var)
- Request builders
- Polling utilities (for async APIs)
- Download helpers
- Error handling

### inference.py Structure

```python
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List

from .{provider}_helper import (
    setup_client,
    create_request,
    poll_status,
    download_result,
)

class AppInput(BaseAppInput):
    prompt: str = Field(description="...")
    # ... other fields

class AppOutput(BaseAppOutput):
    image: File = Field(description="...")
    # ... other fields

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize once per instance."""
        self.client = setup_client()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process each request."""
        result = await process(input_data)
        return AppOutput(...)
```

## Common Patterns

### Secret Keys

| Provider  | Env Var            |
|-----------|--------------------|
| fal.ai    | FAL_KEY            |
| BytePlus  | ARK_API_KEY        |
| Alibaba   | DASHSCOPE_API_KEY  |
| Pruna     | PRUNA_KEY          |

### Polling-Based APIs (Async Workflows)

For APIs that return a task ID and require polling:

```python
async def poll_task_status(
    client,
    task_id: str,
    poll_interval: float = 2.0,
    max_attempts: int = 300,
) -> dict:
    for attempt in range(max_attempts):
        result = client.get_status(task_id)
        if result["status"] == "succeeded":
            return result
        elif result["status"] == "failed":
            raise RuntimeError(f"Task failed: {result.get('error')}")
        await asyncio.sleep(poll_interval)
    raise RuntimeError("Task timed out")
```

### Output Metadata

For pricing/billing purposes, include `OutputMeta`:

```python
from inferencesh import OutputMeta, ImageMeta, VideoMeta

# Image output
output_meta = OutputMeta(
    outputs=[ImageMeta(width=1024, height=1024, count=1)]
)

# Video output
output_meta = OutputMeta(
    outputs=[VideoMeta(width=1920, height=1080, seconds=5.0)]
)

return AppOutput(result=file, output_meta=output_meta)
```

### File Handling

Download results to temp files:

```python
import tempfile
import requests

def download_file(url: str, suffix: str = ".png") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file_path = tmp.name

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path
```

## Deployment Workflow

1. Create app: `infsh app init {model-name}`
2. Symlink helper: `ln -s ../{provider}_helper.py .`
3. Implement `inference.py`
4. Configure `inf.yml` (secrets, category, etc.)
5. Create `README.md` with usage docs
6. Test locally: `infsh run input.json`
7. Deploy: `infsh deploy`
