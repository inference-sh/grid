---
name: managing-secrets
description: Handle API keys and sensitive values in inference.sh apps. Use when adding secrets, accessing environment variables, or securing credentials.
---

# Managing Secrets

Securely access API keys and sensitive values injected at runtime.

## Declaring Secrets

In `inf.yml`:

```yaml
secrets:
  - key: OPENAI_API_KEY
    description: OpenAI API key
    optional: false

  - key: WEBHOOK_SECRET
    description: Optional webhook secret
    optional: true
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `key` | string | Environment variable name |
| `description` | string | Shown to users |
| `optional` | boolean | If false, app won't run without it |

## Accessing Secrets

```python
import os

class App(BaseApp):
    async def setup(self, config):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.client = OpenAI(api_key=api_key)
```

## Common Patterns

### External API Access

```python
from openai import OpenAI

class App(BaseApp):
    async def setup(self, config):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### HuggingFace Token

```yaml
secrets:
  - key: HF_TOKEN
    description: HuggingFace token for gated models
```

```python
from huggingface_hub import snapshot_download

self.model_path = snapshot_download(
    repo_id="meta-llama/Llama-2-7b",
    token=os.environ.get("HF_TOKEN")
)
```

## Tips

- Use specific names (`OPENAI_API_KEY` not `API_KEY`)
- Validate in `setup()`, fail fast
- Never log secret values
