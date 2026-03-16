---
name: optimizing-performance
description: Optimize inference.sh app performance. Use when handling memory, devices, model loading, mixed precision, or flash attention.
---

# Optimizing Performance

Best practices for inference.sh apps.

## Device Detection

**Never hardcode "cuda"** - use accelerate for automatic device detection:

```python
from accelerate import Accelerator

class App(BaseApp):
    async def setup(self, config):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
```

## Model Loading

Use HuggingFace hub for downloads:

```python
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class App(BaseApp):
    async def setup(self, config):
        self.model_path = snapshot_download(
            repo_id="org/model-name",
            resume_download=True,
        )
```

**Avoid**:
- Hardcoded local directories (`local_dir="./models"`)
- Subprocess calls to `huggingface-cli`
- Assuming specific file structures

## Memory Cleanup

```python
import torch, gc

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

async def run(self, input_data):
    result = self.model(input_data)
    cleanup_memory()
    return result
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

## Error Handling

```python
import logging

async def run(self, input_data):
    try:
        result = self.process(input_data)
        return AppOutput(result=result)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise ValueError(f"Failed to process: {str(e)}")
```

## Pre-deploy Checklist

- [ ] All imports work
- [ ] `setup()` loads models
- [ ] `run()` processes test input
- [ ] No hardcoded paths/devices
- [ ] Memory cleaned up
