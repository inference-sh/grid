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

class AppSetup(BaseAppInput):
    """Setup parameters - runs once when config changes"""
    model_id: str = Field(default="gpt2", description="Model to load")

class AppInput(BaseAppInput):
    prompt: str = Field(description="What to generate")
    style: str = Field(default="modern", description="Style")

class AppOutput(BaseAppOutput):
    result: str = Field(description="Generated output")

class App(BaseApp):
    async def setup(self, config: AppSetup):
        """Runs once when worker starts or config changes"""
        self.model = load_model(config.model_id)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Runs for each request"""
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

## Setup Parameters

Use `AppSetup` to define parameters that trigger re-initialization when changed:

```python
class AppSetup(BaseAppInput):
    model_id: str = Field(default="gpt2", description="Model to load")
    precision: str = Field(default="fp16", description="Model precision")

class App(BaseApp):
    async def setup(self, config: AppSetup):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(config.model_id)
```

## File Handling

```python
# Input: auto-downloaded
image_path = input_data.image.path

# Output: auto-uploaded
return AppOutput(image=File(path="/tmp/output.png"))
```

## Multi-Function Apps

Apps can expose multiple functions with different input/output types:

```python
from pydantic import BaseModel

class GreetInput(BaseModel):
    name: str = "World"

class GreetOutput(BaseModel):
    message: str

class ReverseInput(BaseModel):
    text: str

class ReverseOutput(BaseModel):
    reversed_text: str

class App:
    async def run(self, input_data: GreetInput) -> GreetOutput:
        """Default function."""
        return GreetOutput(message=f"Hello, {input_data.name}!")

    async def greet(self, input_data: GreetInput) -> GreetOutput:
        """Custom greeting."""
        return GreetOutput(message=f"Welcome, {input_data.name}!")

    async def reverse(self, input_data: ReverseInput) -> ReverseOutput:
        """Reverse text."""
        return ReverseOutput(reversed_text=input_data.text[::-1])
```

Functions are auto-discovered if they:
- Are public (no `_` prefix)
- Have type hints for input and return
- Use Pydantic models

Call via API with `"function": "reverse"` in the request body.

## The on_cancel Hook

For long-running tasks:

```python
async def on_cancel(self):
    """Called when user cancels - must return quickly"""
    self.cancel_flag = True
    return True
```
