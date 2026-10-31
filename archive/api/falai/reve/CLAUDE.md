# Inference.sh App Development Guide

This guide contains essential knowledge for developing apps on the inference.sh platform, covering best practices, configuration, and common patterns.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Files](#core-files)
3. [Configuration (inf.yml)](#configuration-infyml)
4. [Application Development](#application-development)
5. [Dependencies and Environment](#dependencies-and-environment)
6. [Performance Optimization](#performance-optimization)
7. [Common Patterns](#common-patterns)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

## Project Structure

```
your-app/
├── inference.py          # Main application logic
├── inf.yml              # App configuration variants
├── input.json           # Test input data
├── requirements.txt     # Python dependencies
├── requirements2.txt    # Additional/wheel dependencies (optional)
├── __init__.py         # Python package initialization
└── [additional modules/] # Supporting code
```

## Core Files

### inference.py

The main application file must contain:

```python
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

class AppInput(BaseAppInput):
    # Define input schema with proper Field descriptions
    image: File = Field(description="Input image file")
    prompt: str = Field(description="Text prompt for generation")
    # ... other fields

class AppOutput(BaseAppOutput):
    # Define output schema
    result: File = Field(description="Generated output file")

class App(BaseApp):
    async def setup(self):
        """Initialize models and resources once"""
        # Load models, initialize pipelines
        pass
    
    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process individual requests"""
        # Main inference logic
        return AppOutput(result=output_file)
```

### Key Principles:
- **setup()**: Called once per instance for model loading
- **run()**: Called per request for inference
- Use proper type hints with `File` for file inputs/outputs
- Always provide descriptive `Field(description=...)` for UI generation

## Configuration (inf.yml)

The `inf.yml` file defines app variants with different resource requirements and settings.

### Basic Structure:

```yaml
variants:
  default:
    python_version: "3.10"  # or "3.11", "3.12"
    system_packages: []     # apt packages if needed
    resources:
      ram_gb: 16           # RAM requirement  
      vram_gb: 12          # VRAM requirement
      storage_gb: 50       # Storage requirement
    
  # Additional variants
  lowvram:
    python_version: "3.10"
    resources:
      ram_gb: 8
      vram_gb: 6
      storage_gb: 30
      
  cpu_only:
    python_version: "3.10" 
    resources:
      ram_gb: 8
      vram_gb: 0    # CPU-only
      storage_gb: 20
```

### Advanced Configuration:

```yaml
variants:
  default:
    python_version: "3.10"
    system_packages:
      - ffmpeg           # For video processing
      - libsndfile1      # For audio processing  
    resources:
      ram_gb: 32
      vram_gb: 24
      storage_gb: 100
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      TRANSFORMERS_CACHE: "/tmp/cache"
```

### Resource Guidelines:
- **RAM**: 2-4GB base + model sizes + working memory
- **VRAM**: Model parameters × precision + activation memory + batch overhead
- **Storage**: Models + dependencies + working space (typically 20-100GB)

### Variant Strategy:
- `default`: Best quality/performance
- `lowvram`: Reduced VRAM for smaller GPUs
- `cpu_only`: CPU fallback (much slower)
- `highres`: Higher resolution/quality variants

## Application Development

### Device Management

**❌ Never hardcode "cuda":**
```python
# BAD
device = "cuda"
model.to("cuda")
```

**✅ Use accelerate library:**
```python
from accelerate import Accelerator

class App(BaseApp):
    async def setup(self):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device  # Auto-detects best device
        
        # For models that need device_id as integer
        if hasattr(self.device, 'index') and self.device.index is not None:
            self.device_id = self.device.index
        else:
            self.device_id = 0 if str(self.device) == 'cuda' else 'cpu'
```

### Model Loading Best Practices

**✅ Use proper HuggingFace downloads:**
```python
import os
from huggingface_hub import hf_hub_download, snapshot_download

# Enable faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

async def setup(self):
    # Download entire repositories
    model_dir = snapshot_download("org/model-name")
    
    # Download specific files
    config_file = hf_hub_download("org/model-name", "config.json")
```

**❌ Avoid subprocess calls:**
```python
# BAD
subprocess.run(["huggingface-cli", "download", "model"])
```

### Import Path Management

For complex projects with subdirectories, ensure imports work in production:

```python
# In main __init__.py files, make modules globally available
import sys
current_module = sys.modules[__name__]
sys.modules['your_module_name'] = current_module

# In submodule __init__.py files
from . import submodule
from .submodule import *
```

### Flash Attention Setup

**✅ Proper Flash Attention configuration:**
```python
import os

# Force flash attention globally
os.environ["TRANSFORMERS_ATTENTION_TYPE"] = "flash_attention_2"

# For specific models
model = AutoModel.from_pretrained(
    "model-name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16  # Flash attention requires fp16/bf16
)
```

## Dependencies and Environment

### requirements.txt

```txt
# Core inference.sh requirements
pydantic >= 2.0.0
inferencesh

# ML frameworks with specific versions for stability
# Prefer torch >= 2.7.1 for new GPU support (unless project specifies otherwise)
torch>=2.7.1
torchvision>=0.22.0
torchaudio>=2.7.1

# Alternative: pin to specific stable version if compatibility issues arise
# torch==2.6.0
# torchvision==0.21.0  
# torchaudio==2.6.0

# Acceleration libraries
accelerate>=1.1.1
transformers>=4.49.0

# Utility libraries
numpy>=1.23.5,<2
pillow
einops
```

### requirements2.txt (Optional)

For wheel URLs or special packages:

```txt
# Pre-built wheels for better compatibility
https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Packages that need special handling
git+https://github.com/user/repo.git@branch
```

### Python Version Selection

- **Python 3.10**: Most stable, best package compatibility
- **Python 3.11**: Good performance, some package issues
- **Python 3.12**: Latest features, potential compatibility issues

Choose based on your dependencies' support.

### PyTorch Version Strategy

**Default recommendation: Use PyTorch >= 2.7.1**
- Adds support for newer GPU architectures
- Better performance optimizations
- Enhanced compatibility with modern hardware

**Only pin to older versions when:**
- Project explicitly requires specific version
- Compatibility issues with dependencies
- Model weights incompatible with newer versions

```txt
# Preferred (unless project specifies otherwise)
torch>=2.7.1

# Only if compatibility issues
torch==2.6.0  # or other specific version
```

## Performance Optimization

### Memory Management

```python
import torch
import gc

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Use after inference or model loading
```

### Model Precision

```python
# Use mixed precision for memory efficiency
model = model.to(dtype=torch.bfloat16)  # or torch.float16

# Enable automatic mixed precision
from torch.amp import autocast
with autocast('cuda'):
    output = model(input)
```

### Batch Processing

```python
async def run(self, input_data: AppInput, metadata) -> AppOutput:
    # Process multiple items efficiently
    batch_size = 4
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = self.model(batch)
        results.extend(batch_results)
```

## Common Patterns

### File Handling

```python
import tempfile
import shutil
from pathlib import Path

async def run(self, input_data: AppInput, metadata) -> AppOutput:
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        output_path = tmp.name
    
    # Process and save
    result = process(input_data.image.path)
    save_image(result, output_path)
    
    return AppOutput(image=File(path=output_path))
```

### Configuration Classes

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "default-model"
    batch_size: int = 1
    precision: str = "fp16"
    
class App(BaseApp):
    async def setup(self):
        self.config = ModelConfig()
        # Use self.config throughout
```

### Error Handling

```python
import logging

async def run(self, input_data: AppInput, metadata) -> AppOutput:
    try:
        result = self.process(input_data)
        return AppOutput(result=result)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise ValueError(f"Failed to process input: {str(e)}")
```

## Deployment

### Commands

```bash
# Generate example input file (first time setup)
infsh run
# This creates example.input.json with your schema

# Copy and customize for testing
cp example.input.json input.json
# Edit input.json with your test parameters

# Test locally with your input
infsh run input.json

# Deploy to production
infsh deploy
```

### Input File Generation

When you first run `infsh run` without specifying an input file:
- It automatically generates `example.input.json` based on your `AppInput` schema
- Contains all fields with default/null values
- Shows the exact structure expected by your app

**Workflow:**
1. Run `infsh run` to generate `example.input.json`
2. Copy it to `input.json`: `cp example.input.json input.json`
3. Edit `input.json` with your actual test data
4. Test with `infsh run input.json`
5. Deploy with `infsh deploy`

### Validation

The platform validates:
- Python syntax and imports
- Input/output schema consistency  
- Resource requirements
- File size limits (typically ~100MB)

### Pre-deployment Checklist

- [ ] All imports work correctly
- [ ] `setup()` loads models successfully
- [ ] `run()` processes test input
- [ ] Resource requirements are realistic
- [ ] No hardcoded paths or devices
- [ ] Proper error handling
- [ ] File cleanup (delete temp files)

## Troubleshooting

### Common Issues

**Import Errors in Production:**
- Add proper `__init__.py` files
- Make modules available via `sys.modules`
- Use relative imports consistently

**Memory Issues:**
- Reduce batch sizes
- Use gradient checkpointing: `model.gradient_checkpointing_enable()`
- Clear cache regularly: `torch.cuda.empty_cache()`

**Device Errors:**
- Never hardcode "cuda"
- Always check device availability
- Use accelerate for device management

**Model Loading Errors:**
- Pin dependency versions
- Use `local_files_only=True` after downloading
- Handle missing files gracefully

### Debug Techniques

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Device: {self.device}")
logging.info(f"Model loaded: {type(self.model)}")

# Memory debugging
def log_memory():
    if torch.cuda.is_available():
        logging.info(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
```

### Production Differences

Production environment differs from local:
- Different directory structure
- Import paths may change
- Resource constraints
- Network access patterns

Always test thoroughly before deployment.

## Best Practices Summary

1. **Use accelerate** for device management
2. **Pin dependency versions** for stability
3. **Enable HF Transfer** for faster downloads
4. **Use proper error handling** with informative messages
5. **Clean up temporary files** after processing
6. **Test locally first** before deployment
7. **Use appropriate precision** (fp16/bf16) for memory efficiency
8. **Implement proper logging** for debugging
9. **Handle edge cases** gracefully
10. **Document your input/output schemas** clearly

This guide should be referenced for every inference.sh app development project to ensure consistency and avoid common pitfalls.