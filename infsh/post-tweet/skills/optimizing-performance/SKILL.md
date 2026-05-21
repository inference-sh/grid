---
name: optimizing-performance
description: Optimize inference.sh app performance. Use when handling memory, devices, model loading, mixed precision, or flash attention.
---

# Optimizing Performance

Best practices for inference.sh apps.

## Device Detection

**Never hardcode "cuda"**:

```python
from accelerate import Accelerator
self.device = Accelerator().device
```

## Model Loading

```python
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
model_path = snapshot_download(repo_id="org/model", resume_download=True)
```

## Memory Cleanup

```python
import torch, gc

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
```

## Mixed Precision

```python
model = model.to(dtype=torch.bfloat16)

# Or with autocast
from torch.amp import autocast
with autocast('cuda'):
    output = model(input)
```

## Flash Attention

```python
model = AutoModel.from_pretrained(
    "model-name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

## Pre-deploy Checklist

- [ ] All imports work
- [ ] `setup()` loads models
- [ ] `run()` processes test input
- [ ] No hardcoded paths/devices
- [ ] Memory cleaned up

📖 **Full docs**: [inference.sh/docs/extend/best-practices](https://inference.sh/docs/extend/best-practices)
