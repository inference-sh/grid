---
name: debugging-issues
description: Debug and troubleshoot inference.sh apps. Use when facing import errors, CUDA issues, memory problems, or deployment failures.
---

# Debugging Issues

Common issues and solutions for inference.sh apps.

## Import Errors

### "ModuleNotFoundError" in Production

1. Add `__init__.py` files to all packages

2. Add current directory to Python path:
```python
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

3. For local packages, use editable installs in requirements.txt:
```txt
-e ./local_package_directory
```

## Memory Issues

### CUDA Out of Memory

1. Reduce batch size
2. Use `torch.float16` or `bfloat16`
3. `model.gradient_checkpointing_enable()`
4. `torch.cuda.empty_cache()` after requests
5. Increase `vram` in inf.yml

### Memory Leaks

Clean up after each request:

```python
import gc, torch

async def run(self, input_data):
    result = self.process(input_data)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result
```

## Device Errors

### "Expected all tensors to be on the same device"

Ensure all tensors are on the same device:

```python
input_tensor = input_tensor.to(self.device)
```

### "CUDA not available"

1. Check `inf.yml` GPU requirements:
```yaml
resources:
  gpu:
    count: 1
    vram: 24  # 24GB
```

2. Use device detection:
```python
from accelerate import Accelerator
device = Accelerator().device
```

## Model Loading Errors

### "Token required for gated model"

Add HF_TOKEN to secrets:
```yaml
secrets:
  - key: HF_TOKEN
    description: HuggingFace token for gated models
```

### "File not found" After Download

Don't assume file paths:
```python
model_path = snapshot_download(repo_id="org/model")
config_path = os.path.join(model_path, "config.yaml")
if os.path.exists(config_path):
    # Load config
```

## File Path Issues

### Temporary Files Deleted Too Early

Use `delete=False`:
```python
with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
    output_path = tmp.name
```

### Path Separators

Use `os.path.join`:
```python
# Good
path = os.path.join("models", "config", "settings.json")
```

## Dependency Issues

### Version Conflicts

Pin compatible versions:
```txt
torch==2.6.0
numpy>=1.23.5,<2
```

## Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

async def setup(self, config):
    logging.debug(f"Config: {config}")
    logging.info("Starting model load...")
```
