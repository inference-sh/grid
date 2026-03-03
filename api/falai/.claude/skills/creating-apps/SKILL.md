---
name: building-inferencesh-apps
description: Build and deploy applications on inference.sh. Use when getting started, understanding the platform, or needing an overview of inference.sh development.
---

# Inference.sh App Development

Build and deploy applications on the inference.sh platform.

## CLI Installation

```bash
curl -fsSL https://cli.inference.sh | sh
```

```bash
infsh update   # Update CLI
infsh login    # Authenticate
infsh me       # Check current user
```

## Quick Start

```bash
infsh app init my-app    # Create app
infsh app test           # Test locally
infsh app deploy         # Deploy
```

## App Structure

```python
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppSetup(BaseAppInput):
    """Setup parameters - triggers re-init when changed"""
    model_id: str = Field(default="gpt2", description="Model to load")

class AppInput(BaseAppInput):
    prompt: str = Field(description="Input prompt")

class AppOutput(BaseAppOutput):
    result: str = Field(description="Output result")

class App(BaseApp):
    async def setup(self, config: AppSetup):
        """Runs once when worker starts or config changes"""
        self.model = load_model(config.model_id)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Default function - runs for each request"""
        return AppOutput(result="done")

    async def unload(self):
        """Cleanup on shutdown"""
        pass

    async def on_cancel(self):
        """Called when user cancels - for long-running tasks"""
        return True


# Multi-function apps: add more functions with type hints
class OtherInput(BaseModel):
    text: str

class OtherOutput(BaseModel):
    result: str

class App(BaseApp):
    async def run(self, input_data: AppInput) -> AppOutput:
        """Default function"""
        return AppOutput(result="done")

    async def other_function(self, input_data: OtherInput) -> OtherOutput:
        """Additional function - called via API with function="other_function" """
        return OtherOutput(result=input_data.text.upper())
```

## Related Skills

| Skill | Use When |
|-------|----------|
| [using-the-cli](using-the-cli/) | Running CLI commands for apps and agents |
| [writing-app-logic](writing-app-logic/) | Creating inference.py |
| [configuring-resources](configuring-resources/) | Setting up inf.yml |
| [managing-secrets](managing-secrets/) | Handling API keys |
| [using-oauth-integrations](using-oauth-integrations/) | Google Sheets, Drive |
| [tracking-usage](tracking-usage/) | Output metadata for billing |
| [handling-cancellation](handling-cancellation/) | Long-running tasks |
| [optimizing-performance](optimizing-performance/) | Best practices |
| [debugging-issues](debugging-issues/) | Troubleshooting |

## Resources

- **Full Docs**: [inference.sh/docs](https://inference.sh/docs)
- **Examples**: [github.com/inference-sh/grid](https://github.com/inference-sh/grid)
