---
name: debugging-issues
description: Debug and troubleshoot inference.sh apps. Use when facing import errors, CUDA issues, memory problems, or deployment failures.
---

# Debugging Issues

Common issues and solutions for inference.sh apps.

## Import Errors

```python
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

For local packages: `-e ./local_package` in requirements.txt

## CUDA Out of Memory

1. Reduce batch size
2. Use `torch.float16` or `bfloat16`
3. `model.gradient_checkpointing_enable()`
4. `torch.cuda.empty_cache()` after requests
5. Increase `vram` in inf.yml

## Memory Leaks

```python
import gc, torch

async def run(self, input_data):
    result = self.process(input_data)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result
```

## Device Mismatch

```python
input_tensor = input_tensor.to(self.device)
```

## Gated Models

```yaml
secrets:
  - key: HF_TOKEN
    description: HuggingFace token for gated models
```

## Temp Files Deleted Early

```python
tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
```

## Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

📖 **Full docs**: [inference.sh/docs/extend/troubleshooting](https://inference.sh/docs/extend/troubleshooting)
