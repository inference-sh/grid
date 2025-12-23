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

## Working with Submodules and Complex Projects

### Handling Projects with Relative Paths

When working with existing codebases that use relative paths (like config files loaded at module import time), you may need to manage the working directory:

**Problem**: Module imports fail because they load config files with relative paths like `ovi/configs/config.yaml`.

**Solution**: Temporarily change working directory during imports:

```python
import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'SubModule'))

# Change to submodule directory for imports with relative paths
submodule_dir = os.path.join(current_dir, 'SubModule')
original_cwd = os.getcwd()
os.chdir(submodule_dir)

# Import modules that expect relative paths
from submodule.engine import Engine

# Restore working directory
os.chdir(original_cwd)
```

**Also handle in setup() if initialization needs specific working directory:**

```python
async def setup(self, metadata):
    # ... other setup code

    # Change to submodule directory for initialization
    saved_cwd = os.getcwd()
    os.chdir(submodule_dir)

    try:
        # Initialize components that expect relative paths
        self.engine = Engine(config=config)
    finally:
        # Always restore working directory
        os.chdir(saved_cwd)
```

### Automatic Model Weight Downloads

Implement automatic model downloads in `setup()` to avoid manual download steps:

```python
import os
from huggingface_hub import snapshot_download

# Enable faster downloads at module level
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class App(BaseApp):
    def download_model_weights(self, ckpt_dir: str):
        """Download required model weights from HuggingFace."""
        logging.info("Checking and downloading model weights...")

        # Download model 1
        model1_dir = os.path.join(ckpt_dir, "Model1")
        snapshot_download(
            repo_id="org/model1",
            local_dir=model1_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["weights/*", "config.json"],
            resume_download=True  # Resume if interrupted
        )

        # Download model 2
        model2_dir = os.path.join(ckpt_dir, "Model2")
        snapshot_download(
            repo_id="org/model2",
            local_dir=model2_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.safetensors"],
            resume_download=True
        )

        logging.info("All model weights downloaded successfully!")

    async def setup(self, metadata):
        # Set up checkpoint directory
        ckpt_dir = os.path.join(current_dir, 'ckpts')

        # Download weights automatically
        self.download_model_weights(ckpt_dir)

        # Continue with model initialization
        self.model = load_model(ckpt_dir)
```

**Key Benefits:**
- No manual download step required
- `resume_download=True` handles interrupted downloads
- `allow_patterns` downloads only needed files
- Works on both local testing and production deployment

**Required Dependencies:**
```txt
# In requirements.txt
huggingface_hub>=0.20.0
hf_transfer  # For faster downloads when HF_HUB_ENABLE_HF_TRANSFER=1
```

### Flash Attention Setup

Flash Attention 2 requires special handling due to compilation requirements:

**In requirements2.txt** (NOT requirements.txt):
```txt
# Flash Attention 2 - will be compiled from source
flash-attn>=2.7.0
```

**Why requirements2.txt?**
- Flash Attention needs to be compiled against the installed PyTorch version
- Pre-built wheels often have ABI compatibility issues
- Compiling from source ensures compatibility

**In your code:**
```python
# Models will automatically use flash attention if available
# Make sure to handle the case where it's not available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logging.warning("Flash Attention not available, falling back to standard attention")
```

### Complete Dependency Discovery

When integrating existing projects, discover all dependencies systematically:

1. **Check project's requirements.txt** if it exists
2. **Search for import statements** in the codebase:
   ```bash
   grep -r "^import\|^from" your_module/ | cut -d: -f2 | sort -u
   ```
3. **Run the app and fix ImportError iteratively**
4. **Common hidden dependencies**:
   - `ftfy` - for text cleaning
   - `sentencepiece` - for tokenizers
   - `open-clip-torch` - for CLIP models
   - `protobuf` - for model formats
   - `pydub` - for audio processing
   - `easydict` - for config handling

### Large Model Projects Best Practices

For projects with 30GB+ of model weights:

1. **Use allow_patterns to download only needed files**:
   ```python
   snapshot_download(
       repo_id="large-model/repo",
       allow_patterns=["*.safetensors", "config.json"],  # Skip unnecessary files
       resume_download=True
   )
   ```

2. **Enable HF_HUB_ENABLE_HF_TRANSFER for faster downloads**:
   ```python
   os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
   ```
   Don't forget to add `hf_transfer` to requirements.txt!

3. **Use local_dir for persistent storage**:
   ```python
   snapshot_download(
       repo_id="model",
       local_dir="./ckpts/model",  # Persists across runs
       local_dir_use_symlinks=False
   )
   ```

4. **Log download progress**:
   ```python
   logging.info(f"Downloading {repo_id} to {local_dir}")
   snapshot_download(...)
   logging.info(f"✅ {repo_id} downloaded")
   ```

5. **Download variant-specific model files**:
   When your app has multiple variants (e.g., FP8, quantized), download all variant-specific files:
   ```python
   snapshot_download(
       repo_id="org/model",
       local_dir=model_dir,
       allow_patterns=[
           "model.safetensors",           # For default variant
           "model_fp8_e4m3fn.safetensors" # For fp8 variant
       ],
       resume_download=True
   )
   ```
   This ensures all variants work without requiring separate downloads.

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