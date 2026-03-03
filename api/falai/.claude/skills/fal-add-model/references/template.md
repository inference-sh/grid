# Full App Template

Complete template for fal.ai inference.sh apps.

## inference.py

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

from .fal_helper import setup_fal_client, run_fal_model, download_video, download_image

# Suppress noisy httpx polling logs
logging.getLogger("httpx").setLevel(logging.WARNING)


# Example enum for constrained values
class AspectRatio(str, Enum):
    SQUARE = "1:1"
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"
    WIDE = "21:9"


class AppInput(BaseAppInput):
    """Input schema - add Field descriptions for UI generation."""
    prompt: str = Field(description="The text prompt for generation")

    # Optional image for consolidation (text-to-X + image-to-X)
    image: Optional[File] = Field(
        default=None,
        description="Optional input image. If provided, uses image-to-X mode."
    )

    # Example constrained field
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="Output aspect ratio"
    )

    # Example optional parameters
    num_inference_steps: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of inference steps"
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )


class AppOutput(BaseAppOutput):
    """Output schema."""
    video: File = Field(description="The generated video")
    seed: int = Field(description="The seed used for generation")


class App(BaseApp):
    """App implementation."""

    async def setup(self, metadata):
        """Initialize the application - called once per instance."""
        self.logger = logging.getLogger(__name__)

        # Define model endpoints for consolidation
        self.text_model_id = "fal-ai/model/text-to-video"
        self.image_model_id = "fal-ai/model/image-to-video"

        self.logger.info(f"App initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        """Build the request payload for fal.ai."""
        request = {
            "prompt": input_data.prompt,
            "aspect_ratio": input_data.aspect_ratio.value,
            "num_inference_steps": input_data.num_inference_steps,
        }

        # Add seed if provided
        if input_data.seed is not None:
            request["seed"] = input_data.seed

        # Add image URL for image-to-X mode
        if input_data.image:
            request["image_url"] = input_data.image.uri

        return request

    def _get_model_id(self, input_data: AppInput) -> str:
        """Select model endpoint based on input (consolidation logic)."""
        if input_data.image:
            return self.image_model_id
        return self.text_model_id

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run inference - called per request."""
        try:
            setup_fal_client()

            model_id = self._get_model_id(input_data)
            self.logger.info(f"Using model: {model_id}")
            self.logger.info(f"Processing: {input_data.prompt[:100]}...")

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            # Download the generated file
            video_path = download_video(result["video"]["url"], self.logger)

            return AppOutput(
                video=File(path=video_path),
                seed=result["seed"]
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")
```

## inf.yml

```yaml
name: my-fal-app
description: Description of what this app does
category: video

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

## requirements.txt

```txt
pydantic >= 2.0.0
inferencesh
fal-client>=0.4.0
requests>=2.28.0
```

## __init__.py

```python
from .inference import App, AppInput, AppOutput

__all__ = ["App", "AppInput", "AppOutput"]
```

## Consolidation Patterns

### Pattern 1: Text + Image Input (Same Output)

When fal.ai has separate text-to-X and image-to-X endpoints:

```python
# In _get_model_id()
if input_data.image:
    return "fal-ai/model/image-to-video"
return "fal-ai/model/text-to-video"
```

### Pattern 2: Language Variants

When fal.ai has separate endpoints per language (e.g., TTS):

```python
class Language(str, Enum):
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"

# In _get_model_id()
return f"fal-ai/kokoro-tts/{input_data.language.value}"
```

### Pattern 3: Quality/Speed Variants

When fal.ai has pro/turbo variants:

```python
class Quality(str, Enum):
    FAST = "turbo"
    BALANCED = "standard"
    QUALITY = "pro"

# In _get_model_id()
return f"fal-ai/model/{input_data.quality.value}"
```

## Download Helpers

For different output types:

```python
# Video output
video_path = download_video(result["video"]["url"], self.logger)
return AppOutput(video=File(path=video_path))

# Image output
image_path = download_image(result["images"][0]["url"], self.logger)
return AppOutput(image=File(path=image_path))

# Multiple images
from .fal_helper import download_file
images = []
for img in result["images"]:
    path = download_file(img["url"], suffix=".png", logger=self.logger)
    images.append(File(path=path))
return AppOutput(images=images)

# Audio output
audio_path = download_file(result["audio"]["url"], suffix=".mp3", logger=self.logger)
return AppOutput(audio=File(path=audio_path))
```
