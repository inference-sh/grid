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

## Accessing Secrets

```python
import os

async def setup(self, metadata):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required")
    self.client = OpenAI(api_key=api_key)
```

## Tips

- Use specific names (`OPENAI_API_KEY` not `API_KEY`)
- Validate in `setup()`, fail fast
- Never log secret values

📖 **Full docs**: [inference.sh/docs/extend/secrets](https://inference.sh/docs/extend/secrets)
