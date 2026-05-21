---
name: writing-app-logic
description: Write inference.py for inference.sh apps. Use when creating app logic, defining inputs/outputs, handling files, or implementing setup/run/unload methods.
---

# Writing App Logic (inference.py)

The `inference.py` file contains your app's logic with setup, run, and unload methods.

## Structure

```python
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    prompt: str = Field(description="What to generate")
    style: str = Field(default="modern", description="Style")

class AppOutput(BaseAppOutput):
    result: str = Field(description="Generated output")

class App(BaseApp):
    async def setup(self, metadata):
        """Runs once when worker starts"""
        pass
    
    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Runs for each request"""
        metadata.log("Processing...")
        return AppOutput(result="done")
    
    async def unload(self):
        """Cleanup on shutdown"""
        pass
```

## Field Types

| Type | Usage |
|------|-------|
| `str`, `int`, `float`, `bool` | Basic types |
| `File` | File upload/output (`.path` for local path) |
| `Optional[T]` | Nullable |
| `List[T]` | Array |
| `Literal["a", "b"]` | Enum dropdown |

## File Handling

```python
# Input: auto-downloaded
image_path = input_data.image.path

# Output: auto-uploaded
return AppOutput(image=File(path="/tmp/output.png"))
```

📖 **Full docs**: [inference.sh/docs/extend/app-code](https://inference.sh/docs/extend/app-code)
