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

### Actual Platform Structure:

**IMPORTANT**: The inference.sh platform uses a specific YAML structure. Don't follow generic documentation - use this exact format:

```yaml
name: your-app-name
description: Brief description of your app
category: video  # or image, text, etc.
is_public: false
images:
    card: ""
    thumbnail: ""
    banner: ""
metadata: {}
variants:
    default:
        name: default
        order: 0
        resources:
            gpu:
                count: 1
                vram: 24000000000  # VRAM in bytes (24GB)
                type: any
            ram: 32000000000       # RAM in bytes (32GB)
        env:
            HF_HUB_ENABLE_HF_TRANSFER: "1"
            SAFETENSORS_FAST_GPU: "1"
        python: "3.10"
        
    lowvram:
        name: lowvram
        order: 1
        resources:
            gpu:
                count: 1
                vram: 12000000000  # 12GB
                type: any
            ram: 16000000000       # 16GB
        env: {}
        python: "3.10"
        
    cpu_only:
        name: cpu_only
        order: 2
        resources:
            gpu:
                count: 0
                vram: 0
                type: any
            ram: 16000000000
        env: {}
        python: "3.10"
```

### Key inf.yml Requirements:

- **Resources in bytes**: `vram: 24000000000` not `vram_gb: 24`
- **Each variant needs**: `name`, `order`, `resources`, `env`, `python`
- **GPU specification**: Must include `count`, `vram`, `type: any`
- **Environment variables**: Use `env:` not `environment:`
- **Python version**: Use `python:` not `python_version:`
- **Required fields**: `name`, `description`, `category`, `is_public`

### When to Use Multiple Variants:

**✅ Create variants when your model has:**
- **Different resolutions**: 480p vs 720p versions
- **Different VRAM requirements**: Optimized vs full quality modes
- **Different model configurations**: Base model vs LoRA/fine-tuned versions
- **Different backends**: GPU vs CPU implementations
- **Feature toggles**: Different capabilities enabled/disabled

**❌ Don't create variants if:**
- Your model has consistent behavior and requirements
- No meaningful configuration differences exist
- All users would use the same settings

### Variant Implementation Pattern:

**IMPORTANT**: Variants are read from `metadata.app_variant`, NOT from `os.environ`!

```python
async def setup(self, metadata):
    # CORRECT: Read variant from metadata
    variant = getattr(metadata, "app_variant", "default")
    
    # Map variant to configuration
    if variant == "high_quality":
        model_type = "high-res-model"
        offload = False
    elif variant == "low_vram":
        model_type = "quantized-model"
        offload = True
    else:
        model_type = "default-model"
        offload = False
    
    # Initialize based on variant
    self.pipeline = SomePipeline(model_type, offload=offload)
```

**❌ WRONG - Don't do this:**
```python
# BAD - variants are NOT read from environment variables
variant = os.environ.get("VARIANT_TYPE", "default")
model_type = os.environ.get("MODEL_TYPE", "default")
```

**✅ CORRECT - Read from metadata:**
```python
# GOOD - variants come from metadata.app_variant
variant = getattr(metadata, "app_variant", "default")
```

### Real-World Variant Example:

Here's a complete example showing how to properly implement variants:

**inf.yml:**
```yaml
variants:
    default:
        name: flux-dev  # The actual name can be different from the key
        order: 0
        resources:
            gpu:
                count: 1
                vram: 24000000000  # 24GB
                type: any
            ram: 32000000000       # 32GB
        env:
            HF_HUB_ENABLE_HF_TRANSFER: "1"
        python: "3.10"
        
    flux-dev-fp8:
        name: flux-dev-fp8
        order: 1
        resources:
            gpu:
                count: 1
                vram: 16000000000  # 16GB
                type: any
            ram: 24000000000       # 24GB
        env:
            HF_HUB_ENABLE_HF_TRANSFER: "1"
        python: "3.10"
```

**inference.py:**
```python
async def setup(self, metadata):
    # Get variant from metadata (this is the variant KEY from inf.yml)
    variant = getattr(metadata, "app_variant", "default")
    
    # Map variant KEY to model configuration
    if variant == "default":
        model_type = "flux-dev"  # Use the actual model name
        enable_offload = False
    elif variant == "flux-dev-fp8":
        model_type = "flux-dev-fp8"
        enable_offload = True  # Enable for lower VRAM
    else:
        # Fallback
        model_type = "flux-dev"
        enable_offload = False
        
    # Initialize pipeline based on variant
    self.pipeline = MyPipeline(
        model_type=model_type,
        offload=enable_offload
    )
```

### Key Variant Rules:

1. **Variant key**: The variant KEY (not name) from inf.yml becomes `metadata.app_variant`
2. **Default key required**: At least one variant key must be `default` 
3. **Key vs name**: The variant key can be different from the `name` field
4. **No environment variables**: Don't use `os.environ` to pass variant information
5. **Order matters**: Lower `order` values are preferred/default
6. **Resource specification**: Only specify `resources` and `env` for variants that need them
7. **Fallback handling**: Always provide a fallback for unknown variants

**Important**: `metadata.app_variant` contains the YAML key (e.g., "default", "flux-dev-fp8"), not the `name` field value!

### Single Variant Example:

For models with consistent behavior, use only a `default` variant:

```yaml
variants:
    default:
        name: default
        order: 0
        resources:
            gpu:
                count: 1
                vram: 24000000000
                type: any
            ram: 32000000000
        env:
            HF_HUB_ENABLE_HF_TRANSFER: "1"
        python: "3.10"
        
    low_vram:
        name: low_vram
        order: 1
        resources:
            gpu:
                count: 1
                vram: 12000000000
                type: any
            ram: 16000000000
        env:
            HF_HUB_ENABLE_HF_TRANSFER: "1"
        python: "3.10"
```

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
from huggingface_hub import snapshot_download

# Enable faster downloads globally at module level
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.repo_id = "org/model-name"

    async def setup(self, metadata):
        # Download to HuggingFace cache (not custom local_dir)
        # This enables proper caching across runs
        self.model_path = snapshot_download(
            repo_id=self.repo_id,
            resume_download=True,  # Always resume when possible
            local_files_only=False,  # Allow network downloads
        )
        
        # Load model from cache location
        config_path = os.path.join(self.model_path, "config.yaml")
        # ... continue with model loading
```

**❌ Avoid these patterns:**
```python
# BAD - hardcoded local directories
model_dir = snapshot_download("org/model-name", local_dir="./models")

# BAD - subprocess calls
subprocess.run(["huggingface-cli", "download", "model"])

# BAD - assuming file structure
config_path = "./config.yaml"  # May not exist
```

### HuggingFace Cache Strategy

The platform uses HuggingFace's built-in cache system:
- **Use `snapshot_download()` without `local_dir`** - lets HF handle caching
- **Models download to**: `/mnt/raid/cache/huggingface/hub/models--org--model-name/`
- **Files are symlinked**: Check actual file structure after download
- **Cache persists**: Subsequent runs reuse downloaded models

### Import Path Management

For complex projects with subdirectories, ensure imports work in production:

**✅ Add current directory to Python path:**
```python
# At the top of inference.py
import os
import sys

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can import local modules
from your_local_module import something
```

**✅ Create __init__.py files:**
```python
# In main __init__.py files, make modules globally available
import sys
current_module = sys.modules[__name__]
sys.modules['your_module_name'] = current_module

# In submodule __init__.py files
from . import submodule
from .submodule import *
```

### Local Package Dependencies

When your app depends on locally cloned packages:

**✅ Use editable installs in requirements.txt:**
```txt
# For locally cloned packages
-e ./local_package_directory

# NOT this - will fail
git+./local_package
./local_package
```

**Production Import Issues:**
- Always test imports work with `infsh run`
- Production environment may have different import behavior
- Add debug logging to track import issues
- Use absolute imports when possible

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

## LoRA Integration for Diffusers Pipelines

When working with diffusers-based models that support LoRA (Low-Rank Adaptation), follow this complete implementation pattern to add robust LoRA support to your inference apps.

### LoRA Configuration Schema

First, add the LoRA configuration model and input schema:

```python
from pydantic import BaseModel, Field
from typing import Optional

class LoraConfig(BaseModel):
    adapter_name: str = Field(description="Name for the LoRA adapter.")
    lora_file: File = Field(description="LoRA weights file (.safetensors)")
    lora_multiplier: float = Field(default=1.0, ge=0.0, le=10.0, description="Multiplier for the LoRA effect")

class AppInput(BaseAppInput):
    # ... your existing fields
    loras: Optional[list[LoraConfig]] = Field(default=None, description="List of LoRA configs to apply")
```

### LoRA Support Functions

Add these comprehensive LoRA helper functions to handle various source types:

```python
import requests
import re
import os
from datetime import datetime
import logging

def get_civit_download_url(model_id):
    """Get download URL for CivitAI model with optional token auth."""
    civitai_token = os.environ.get('CIVITAI_TOKEN')
    if civitai_token:
        url = f"https://civitai.com/models/{model_id}&token={civitai_token}"
    else:
        url = f"https://civitai.com/models/{model_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    versions = data["modelVersions"]

    def get_date(item):
        date_str = item.get('updatedAt') or item.get('publishedAt') or item.get('createdAt') or ''
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return datetime.fromisoformat(date_str)

    # Prefer Qwen-compatible versions, fallback to latest
    sorted_data = sorted(versions, key=get_date, reverse=True)
    qwen_versions = [v for v in sorted_data if v.get('baseModel', '').lower().startswith('qwen')]
    latest_version = qwen_versions[0] if qwen_versions else sorted_data[0]
    return latest_version["downloadUrl"]

def download_model_data(model_id, downloadUrl, targetFolder):
    """Download LoRA file with CivitAI token support."""
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    filePath = os.path.join(targetFolder, f"{model_id}.safetensors")
    if not os.path.isfile(filePath):
        civitai_token = os.environ.get('CIVITAI_TOKEN')
        if civitai_token:
            print("Using CivitAI token from environment variable CIVITAI_TOKEN")
        else:
            print("WARNING: No CivitAI token found in environment variable CIVITAI_TOKEN")

        if 'civitai.com' in downloadUrl and civitai_token:
            if '?' in downloadUrl:
                downloadUrl = f"{downloadUrl}&token={civitai_token}"
            else:
                downloadUrl = f"{downloadUrl}?token={civitai_token}"
        
        response = requests.get(downloadUrl)
        response.raise_for_status()
        with open(filePath, 'wb') as file:
            file.write(response.content)
    return filePath

def load_lora_adapter(pipeline, lora_source, adapter_name="lora", lora_multiplier=1.0):
    """
    Load LoRA adapter with fallback strategy: transformer first, then pipeline.
    Supports: local files, HuggingFace URLs, direct URLs, CivitAI models.
    """
    if not lora_source:
        return []
    
    def _load_single_lora(load_kwargs, base_adapter_name: str):
        # Try loading on transformer first (more direct LoRA support)
        try:
            transformer = getattr(pipeline, 'transformer', None)
            if transformer and hasattr(transformer, 'load_lora_weights'):
                print(f"🔄 Loading LoRA adapter '{base_adapter_name}' onto transformer...")
                load_kwargs["adapter_name"] = base_adapter_name
                transformer.load_lora_weights(**load_kwargs)
                print(f"✅ LoRA adapter '{base_adapter_name}' loaded successfully onto transformer")
                logging.info(f"Loaded LoRA adapter {base_adapter_name} onto transformer")
                return base_adapter_name
        except Exception as e:
            print(f"⚠️ Failed to load LoRA on transformer: {e}, trying pipeline...")
            logging.warning(f"Failed to load LoRA {base_adapter_name} on transformer: {e}")
        
        # Fallback to pipeline (diffusers standard approach)
        try:
            if hasattr(pipeline, 'load_lora_weights'):
                print(f"🔄 Loading LoRA adapter '{base_adapter_name}' onto pipeline...")
                load_kwargs["adapter_name"] = base_adapter_name
                pipeline.load_lora_weights(**load_kwargs)
                print(f"✅ LoRA adapter '{base_adapter_name}' loaded successfully onto pipeline")
                logging.info(f"Loaded LoRA adapter {base_adapter_name} onto pipeline")
                return base_adapter_name
            else:
                print(f"❌ Neither transformer nor pipeline support load_lora_weights method")
        except Exception as e:
            print(f"❌ Failed to load LoRA '{base_adapter_name}' on pipeline: {e}")
            logging.error(f"Failed to load LoRA {base_adapter_name}: {e}")
        return None
    
    # Handle different source types:
    
    # 1. Local file path
    if isinstance(lora_source, str) and os.path.isfile(lora_source):
        load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_source}
        created = _load_single_lora(load_kwargs, adapter_name)
        return [created] if created else []

    # 2. HuggingFace blob URL (https://huggingface.co/user/repo/blob/main/file.safetensors)
    if isinstance(lora_source, str) and "huggingface.co" in lora_source and "/blob/" in lora_source:
        parts = lora_source.split('/')
        if len(parts) >= 7 and 'huggingface.co' in parts and 'blob' in parts:
            repo_start = parts.index('huggingface.co') + 1
            blob_index = parts.index('blob')
            repo_id = '/'.join(parts[repo_start:blob_index])
            weight_name = '/'.join(parts[blob_index + 2:])
            load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
            created = _load_single_lora(load_kwargs, adapter_name)
            return [created] if created else []
    
    # 3. HuggingFace resolve URL (https://huggingface.co/user/repo/resolve/main/file.safetensors)
    elif isinstance(lora_source, str) and "huggingface.co" in lora_source and "/resolve/" in lora_source:
        parts = lora_source.split('/')
        if len(parts) >= 7 and 'huggingface.co' in parts and 'resolve' in parts:
            repo_start = parts.index('huggingface.co') + 1
            resolve_index = parts.index('resolve')
            repo_id = '/'.join(parts[repo_start:resolve_index])
            weight_name = '/'.join(parts[resolve_index + 2:])
            load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
            created = _load_single_lora(load_kwargs, adapter_name)
            return [created] if created else []
    
    # 4. HuggingFace repository (user/repo or user/repo/file.safetensors)
    elif isinstance(lora_source, str) and "/" in lora_source and not lora_source.startswith('http') and "civitai.com" not in lora_source:
        parts = lora_source.split('/')
        if len(parts) == 2:  # Just repo
            load_kwargs = {"repo_id": lora_source}
            created = _load_single_lora(load_kwargs, adapter_name)
            return [created] if created else []
        elif len(parts) > 2:  # Repo with file path
            repo_id = '/'.join(parts[:2])
            weight_name = '/'.join(parts[2:])
            load_kwargs = {"repo_id": repo_id, "weight_name": weight_name}
            created = _load_single_lora(load_kwargs, adapter_name)
            return [created] if created else []
    
    # 5. Direct .safetensors URL
    elif isinstance(lora_source, str) and lora_source.endswith('.safetensors') and lora_source.startswith('http'):
        lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)
        
        model_id = os.path.splitext(os.path.basename(lora_source))[0]
        lora_path = download_model_data(model_id, lora_source, lora_dir)
        load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_path}
        created = _load_single_lora(load_kwargs, adapter_name)
        return [created] if created else []
    
    # 6. CivitAI model URL (https://civitai.com/models/12345)
    elif isinstance(lora_source, str) and "civitai.com" in lora_source:
        match = re.search(r"/models/(\d+)", lora_source)
        if not match:
            raise ValueError("Could not extract model ID from CivitAI URL")
        
        model_id = match.group(1)
        download_url = get_civit_download_url(model_id)
        if not download_url:
            raise RuntimeError(f"No download URL found for CivitAI model {model_id}")
        
        lora_dir = "loras" if os.path.isdir("loras") else "/tmp/loras"
        lora_path = download_model_data(model_id, download_url, lora_dir)
        load_kwargs = {"pretrained_model_name_or_path_or_dict": lora_path}
        created = _load_single_lora(load_kwargs, adapter_name)
        return [created] if created else []
    
    else:
        raise ValueError(f"Unsupported LoRA source format: {lora_source}")
```

### App Integration Pattern

Add LoRA tracking to your App class:

```python
class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and LoRA tracking."""
        # ... your existing setup code
        
        # Initialize LoRA tracking
        self.loaded_loras = {}  # adapter_name -> (source, multiplier, created_names)
        
        # ... rest of setup
        
    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process request with LoRA handling."""
        # ... your existing pre-processing
        
        # Handle LoRA adapters
        loras = getattr(input_data, "loras", None) or []
        requested_by_name = {l.adapter_name: l for l in loras}

        # Unload adapters that changed or are no longer needed
        for adapter_name in list(self.loaded_loras.keys()):
            previous_source, previous_mult, previous_created = self.loaded_loras[adapter_name]
            found = requested_by_name.get(adapter_name)
            if (
                found is None
                or previous_source != found.lora_file.path
                or previous_mult != found.lora_multiplier
            ):
                for created_name in previous_created:
                    # Try deleting from transformer first
                    deleted = False
                    if hasattr(self.pipeline.transformer, 'delete_adapters'):
                        try:
                            self.pipeline.transformer.delete_adapters(created_name)
                            deleted = True
                            print(f"🗑️ Unloaded LoRA adapter '{created_name}' from transformer")
                        except Exception as e:
                            print(f"⚠️ Failed to delete LoRA from transformer: {e}, trying pipeline...")
                    
                    # Fallback to pipeline deletion
                    if not deleted and hasattr(self.pipeline, 'delete_adapters'):
                        try:
                            self.pipeline.delete_adapters(created_name)
                            deleted = True
                            print(f"🗑️ Unloaded LoRA adapter '{created_name}' from pipeline")
                        except Exception as e:
                            print(f"❌ Failed to delete LoRA from pipeline: {e}")
                    
                    if deleted:
                        logging.info(f"Unloaded previous LoRA adapter: {created_name}")
                del self.loaded_loras[adapter_name]

        # Load new/changed adapters
        for lora in loras:
            needs_load = (
                lora.adapter_name not in self.loaded_loras
                or self.loaded_loras[lora.adapter_name][0] != lora.lora_file.path
                or self.loaded_loras[lora.adapter_name][1] != lora.lora_multiplier
            )
            if needs_load:
                created_names = load_lora_adapter(
                    self.pipeline,
                    lora.lora_file.path,
                    lora.adapter_name,
                    lora.lora_multiplier,
                )
                if created_names:
                    self.loaded_loras[lora.adapter_name] = (
                        lora.lora_file.path,
                        lora.lora_multiplier,
                        created_names,
                    )

        # Activate all requested adapters
        active_adapters = []
        adapter_weights = []
        for lora in loras:
            if lora.adapter_name in self.loaded_loras:
                created_names = self.loaded_loras[lora.adapter_name][2]
                for created_name in created_names:
                    active_adapters.append(created_name)
                    adapter_weights.append(lora.lora_multiplier)

        # Apply adapter activation with fallback strategy
        if active_adapters:
            print(f"🎯 Activating LoRA adapters: {active_adapters} with weights: {adapter_weights}")
            
            # Handle single adapter activation (diffusers standard API)
            if len(active_adapters) == 1:
                adapter_name = active_adapters[0]
                adapter_weight = adapter_weights[0]
                
                # Try activating on pipeline first (diffusers standard)
                activated = False
                if hasattr(self.pipeline, 'set_adapters'):
                    try:
                        self.pipeline.set_adapters(adapter_name, adapter_weights=adapter_weight)
                        print(f"✅ LoRA adapter '{adapter_name}' activated on pipeline successfully")
                        activated = True
                    except Exception as e:
                        print(f"⚠️ Failed to activate LoRA on pipeline: {e}, trying transformer...")
                
                # Fallback to transformer activation
                if not activated and hasattr(self.pipeline.transformer, 'set_adapters'):
                    try:
                        self.pipeline.transformer.set_adapters(adapter_name, adapter_weights=adapter_weight)
                        print(f"✅ LoRA adapter '{adapter_name}' activated on transformer successfully")
                        activated = True
                    except Exception as e:
                        print(f"❌ Failed to activate LoRA on transformer: {e}")
                
                if not activated:
                    print(f"⚠️ Found adapter '{adapter_name}' but neither pipeline nor transformer support set_adapters")
            
            # Handle multiple adapters (list format - may work for some models)
            else:
                # Try activating on transformer first
                activated = False
                if hasattr(self.pipeline.transformer, 'set_adapters'):
                    try:
                        self.pipeline.transformer.set_adapters(active_adapters, adapter_weights=adapter_weights)
                        print(f"✅ LoRA adapters activated on transformer successfully")
                        activated = True
                    except Exception as e:
                        print(f"⚠️ Failed to activate LoRA on transformer: {e}, trying pipeline...")
                
                # Fallback to pipeline activation
                if not activated and hasattr(self.pipeline, 'set_adapters'):
                    try:
                        self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
                        print(f"✅ LoRA adapters activated on pipeline successfully")
                        activated = True
                    except Exception as e:
                        print(f"❌ Failed to activate LoRA on pipeline: {e}")
                
                if not activated:
                    print(f"⚠️ Found {len(active_adapters)} adapters but neither transformer nor pipeline support set_adapters")
        else:
            print("ℹ️ No custom LoRA adapters to activate")
        
        # ... continue with your inference logic
```

### Required Dependencies

Add these to your `requirements.txt`:

```txt
# Required for LoRA functionality
peft>=0.7.0  # Essential for LoRA adapter support
requests>=2.25.0
python-dateutil>=2.8.0
```

### Environment Variables

Set these for enhanced functionality:

```python
# Enable faster HuggingFace downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Optional: Add CivitAI token for private models
# Set CIVITAI_TOKEN environment variable in inf.yml
```

In your `inf.yml`:

```yaml
env:
    HF_HUB_ENABLE_HF_TRANSFER: "1"
    # CIVITAI_TOKEN: "your-token-here"  # Optional, for private CivitAI models
```

### Usage Examples

Users can provide LoRAs in various formats:

```json
{
  "loras": [
    {
      "adapter_name": "style_lora", 
      "lora_file": {"path": "/path/to/local/lora.safetensors"},
      "lora_multiplier": 0.8
    },
    {
      "adapter_name": "character_lora",
      "lora_file": {"path": "https://huggingface.co/user/repo/resolve/main/lora.safetensors"},
      "lora_multiplier": 1.2
    },
    {
      "adapter_name": "civitai_lora",
      "lora_file": {"path": "https://civitai.com/models/12345"},
      "lora_multiplier": 1.0
    }
  ]
}
```

### Key Benefits

This implementation provides:

- **Multiple source support**: Local files, HuggingFace, CivitAI, direct URLs
- **Robust fallback**: Tries transformer then pipeline loading/activation
- **Dynamic management**: Load/unload LoRAs between requests efficiently
- **Clear feedback**: Visual indicators for loading success/failure
- **Weight control**: Per-adapter multiplier support
- **Error resilience**: Graceful handling of unsupported formats

### Implementation Notes

- LoRAs are loaded per-request and persist between requests for efficiency
- Only changed LoRAs are reloaded (smart caching)
- Clear visual feedback helps debug LoRA loading issues
- Supports both transformer-level and pipeline-level LoRA systems
- Compatible with all diffusers pipeline models that support LoRA

Add this pattern to any diffusers-based inference app to provide comprehensive LoRA support.

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
- Use `sys.path.append()` in inference.py
- Use `-e ./package` for local packages in requirements.txt

**Memory Issues:**
- Reduce batch sizes
- Use gradient checkpointing: `model.gradient_checkpointing_enable()`
- Clear cache regularly: `torch.cuda.empty_cache()`

**Device Errors:**
- Never hardcode "cuda"
- Always check device availability
- Use accelerate for device management

**Model Loading Errors:**
- Don't assume file paths - check what HuggingFace downloads
- Use `os.path.join()` for cross-platform paths
- Handle missing config files gracefully
- Pin dependency versions for stability

**File Path Issues:**
- HuggingFace models may have different file structures than expected
- Always check downloaded directory contents first
- Use `os.path.exists()` before assuming files exist
- Config files might be named differently (e.g., `config.yaml` vs `config.json`)

### Development Workflow Issues

**Testing and Input Files:**
- Run `infsh run` first to generate `example.input.json`
- Copy to `input.json` and edit with real data
- Use cloud URLs for file inputs in testing
- Test thoroughly before deployment

**Environment Variables:**
- Set `os.environ` at module level, not in functions
- Use HF_HUB_ENABLE_HF_TRANSFER for faster downloads
- Check inf.yml `env:` section for runtime variables

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