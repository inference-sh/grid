# fal.ai Apps Documentation

This directory contains inference.sh apps that wrap fal.ai models.

## Directory Structure

```
falai/
├── DOCS.md              # This file
├── fal_helper.py        # Shared helper module (copy into your app folder)
├── seedance-1-5-pro-i2v/
├── seedance-1-5-pro-t2v/
├── flux-2-dev-fal/
├── wan-2-5/
├── wan-2-5-i2v/
└── ...
```

## Quick Start

1. **Create a new app folder:**
   ```bash
   infsh app init my-fal-app
   cd my-fal-app
   ```

2. **Copy the shared helper:**
   ```bash
   cp ../fal_helper.py .
   ```

3. **Update `requirements.txt`:**
   ```txt
   pydantic >= 2.0.0
   inferencesh
   fal-client>=0.4.0
   requests>=2.28.0
   ```

4. **Configure `inf.yml`:**
   ```yaml
   name: my-fal-app
   description: Description of your app
   category: video  # or image, audio, etc.
   
   kernel: python-3.11
   
   resources:
     gpu:
       count: 0
       vram: 0
       type: none
     ram: 4000000000
   
   env: {}
   
   secrets:
     - key: FAL_KEY
       description: fal.ai API key for model access
       optional: false
   ```

## Using `fal_helper.py`

The shared helper provides these utilities:

### `setup_fal_client(api_key=None)`

Configure the fal.ai client with an API key.

```python
from fal_helper import setup_fal_client

# Uses FAL_KEY environment variable
setup_fal_client()

# Or pass explicitly
setup_fal_client("your-api-key")
```

### `run_fal_model(model_id, arguments, logger, with_logs=True)`

Execute a fal.ai model with progress logging.

```python
from fal_helper import run_fal_model

result = run_fal_model(
    model_id="fal-ai/some-model",
    arguments={"prompt": "..."},
    logger=self.logger
)
```

### `download_video(url, logger=None)` / `download_file(url, suffix, logger=None)`

Download generated files to a temporary location.

```python
from fal_helper import download_video

video_path = download_video(result["video"]["url"], self.logger)
return AppOutput(video=File(path=video_path))
```

## Example App Template

```python
"""
My fal.ai App

Description of what this app does.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

# Suppress noisy httpx polling logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class AppInput(BaseAppInput):
    """Input schema."""
    prompt: str = Field(description="The text prompt")
    # Add more fields as needed


class AppOutput(BaseAppOutput):
    """Output schema."""
    video: File = Field(description="The generated video")
    seed: int = Field(description="The seed used")


class App(BaseApp):
    """App implementation."""
    
    async def setup(self, metadata):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        self.model_id = "fal-ai/your-model-endpoint"
        self.logger.info(f"App initialized with model: {self.model_id}")

    def _build_request(self, input_data: AppInput) -> dict:
        """Build the request payload."""
        return {
            "prompt": input_data.prompt,
            # Add more fields
        }

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run inference."""
        try:
            setup_fal_client()
            
            self.logger.info(f"Processing: {input_data.prompt[:100]}...")
            
            request_data = self._build_request(input_data)
            result = run_fal_model(self.model_id, request_data, self.logger)
            
            video_path = download_video(result["video"]["url"], self.logger)
            
            return AppOutput(
                video=File(path=video_path),
                seed=result["seed"]
            )
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")
```

## Common fal.ai Model Endpoints

| Model | Endpoint |
|-------|----------|
| Seedance 1.5 Pro I2V | `fal-ai/bytedance/seedance/v1.5/pro/image-to-video` |
| Seedance 1.5 Pro T2V | `fal-ai/bytedance/seedance/v1.5/pro/text-to-video` |
| Flux 2 | `fal-ai/flux-2` |
| Flux 2 Edit | `fal-ai/flux-2/edit` |
| Wan 2.5 I2V | `fal-ai/wan-25-preview/image-to-video` |

## Tips

1. **No GPU needed**: fal.ai apps run inference remotely, so set `gpu.count: 0` and `gpu.type: none`

2. **Use enums for constrained values**: Define `Enum` classes for parameters with fixed options (resolution, aspect ratio, etc.)

3. **File inputs**: Use `input_data.image.uri` for the URL that fal.ai can access, not `.path`

4. **Error handling**: Wrap the run method in try/except and raise `RuntimeError` with descriptive messages

5. **Logging**: Use `self.logger.info()` for progress updates that help debug issues

## Resources

- [fal.ai Documentation](https://docs.fal.ai)
- [fal.ai Model Playground](https://fal.ai/models)
- [Python Client Docs](https://docs.fal.ai/clients/python)

