#!/bin/bash
#
# or-scaffold.sh - Scaffold an OpenRouter model app from the API
#
# Usage:
#   ./or-scaffold.sh <model-id> [app-dir]
#   ./or-scaffold.sh anthropic/claude-sonnet-4.6
#   ./or-scaffold.sh anthropic/claude-sonnet-4.6 claude-sonnet-46
#
# 1. Runs `belt app init` to create the proper app skeleton
# 2. Fetches model capabilities & pricing from the OpenRouter API
# 3. Overwrites inference.py, inf.yml, __init__.py, requirements.txt
#    with OpenRouter-specific versions based on model capabilities
# 4. Symlinks the shared openrouter.py helper
# 5. Generates MODEL.md with pricing and capability reference
#
# Requires: belt, curl, python3

set -e

MODEL_ID="${1:-}"
APP_DIR="${2:-}"

if [[ -z "$MODEL_ID" ]]; then
    echo "Usage: $0 <model-id> [app-dir]"
    echo "Example: $0 anthropic/claude-sonnet-4.6"
    echo "Example: $0 anthropic/claude-sonnet-4.6 claude-sonnet-46"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Step 1: Derive app dir name from model ID if not provided ---
if [[ -z "$APP_DIR" ]]; then
    APP_DIR=$(echo "$MODEL_ID" | sed 's|.*/||' | sed 's|:.*||' | tr '.' '' | tr -cs 'a-z0-9-' '-' | sed 's/-$//')
fi

# --- Step 2: belt app init (creates the proper skeleton) ---
if [[ -d "$SCRIPT_DIR/$APP_DIR" ]]; then
    echo "Directory $APP_DIR already exists, skipping belt app init"
else
    echo "=== Step 1: belt app init $APP_DIR ==="
    cd "$SCRIPT_DIR"
    belt app init "$APP_DIR"
fi

# --- Step 3: Fetch model from OpenRouter API ---
echo "=== Step 2: Fetching model: $MODEL_ID ==="

curl -s "https://openrouter.ai/api/v1/models" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('data', [])
model = next((m for m in models if m['id'] == '$MODEL_ID'), None)
if not model:
    print('NOT_FOUND')
    similar = [m['id'] for m in models if any(p in m['id'].lower() for p in '$MODEL_ID'.lower().replace('/', ' ').split())]
    for s in similar[:10]:
        print(s)
    sys.exit(0)
json.dump(model, sys.stdout)
" > /tmp/or_model.json

if head -1 /tmp/or_model.json | grep -q "NOT_FOUND"; then
    echo "Error: Model not found: $MODEL_ID"
    echo "Similar models:"
    tail -n +2 /tmp/or_model.json | sed 's/^/  /'
    rm -f /tmp/or_model.json
    exit 1
fi

# --- Step 4: Overlay OpenRouter-specific files ---
echo "=== Step 3: Generating OpenRouter files ==="

python3 - "$SCRIPT_DIR" "$APP_DIR" <<'PYSCRIPT'
import sys, json, os, re

script_dir = sys.argv[1]
app_dir = sys.argv[2]
out_dir = os.path.join(script_dir, app_dir)

with open("/tmp/or_model.json") as f:
    model = json.load(f)
os.remove("/tmp/or_model.json")

# --- Extract model info ---
model_id = model["id"]
name = model.get("name", model_id)
description = model.get("description", "")
context_length = model.get("context_length", 200000)
arch = model.get("architecture", {})
modality = arch.get("modality", "text->text")
input_modalities = arch.get("input_modalities", ["text"])
pricing = model.get("pricing", {})
prompt_price = float(pricing.get("prompt", "0"))
completion_price = float(pricing.get("completion", "0"))
supported_params = model.get("supported_parameters", [])
top_provider = model.get("top_provider", {})
max_completion = top_provider.get("max_completion_tokens", 64000) or 64000

prompt_per_m = prompt_price * 1_000_000
completion_per_m = completion_price * 1_000_000

# --- Determine capabilities from modality ---
capabilities = ["reasoning"]
has_image = "image" in input_modalities
has_file = "file" in input_modalities

if has_image:
    capabilities.append("image")
if has_file:
    capabilities.append("file")

# --- Build mixin imports based on capabilities ---
extra_imports = []
input_mixins = ["LLMInput", "ReasoningCapabilityMixin", "ToolsCapabilityMixin"]
if has_image:
    extra_imports.append("ImageCapabilityMixin")
    input_mixins.append("ImageCapabilityMixin")
if has_file:
    extra_imports.append("FileCapabilityMixin")
    input_mixins.append("FileCapabilityMixin")

all_imports = [
    "LLMInput",
    "LLMOutput",
    "ReasoningCapabilityMixin",
    "ReasoningMixin",
    "ToolsCapabilityMixin",
    "ToolCallsMixin",
] + extra_imports

import_block = ",\n    ".join(all_imports)
mixin_str = ", ".join(input_mixins)

short_desc = description.split(".")[0].strip() + "." if "." in description else description[:200]

# --- Overlay files onto belt-initialized app ---

# inference.py
with open(os.path.join(out_dir, "inference.py"), "w") as f:
    f.write(f'''import os
from typing import AsyncGenerator, List, Optional
from pydantic import Field

from inferencesh import BaseApp, BaseAppOutput
from inferencesh.models.llm import (
    {import_block}
)
from .openrouter import stream_completion
from openai import AsyncOpenAI

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = "{model_id}"


class AppInput({mixin_str}):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default={context_length}, description="The context size for the model.")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""
    images: Optional[List[str]] = None


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        print("OpenRouter client initialization complete!")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized. Call setup() first.")

        async for output in stream_completion(self.client, input_data, DEFAULT_MODEL):
            yield AppOutput(**output)

    async def unload(self):
        self.client = None
''')

# inf.yml
caps_yaml = "\n".join(f"        - {c}" for c in capabilities)
with open(os.path.join(out_dir, "inf.yml"), "w") as f:
    f.write(f'''namespace: openrouter
name: {app_dir}
description: {short_desc}
metadata:
    capabilities:
{caps_yaml}
category: chat
images:
    card: ""
    thumbnail: ""
    banner: ""
env: {{}}
kernel: ""
secrets:
    - key: OPENROUTER_API_KEY
resources:
    gpu:
        count: 0
        vram: 0
        type: none
    ram: 0
''')

# __init__.py
with open(os.path.join(out_dir, "__init__.py"), "w") as f:
    f.write("from .inference import App, AppInput, AppOutput\n")

# requirements.txt
with open(os.path.join(out_dir, "requirements.txt"), "w") as f:
    f.write("pydantic >= 2.0.0\n\n\ninferencesh >= 0.6.30\nopenai >= 1.0.0\n")

# Symlink openrouter.py (remove belt's default if exists)
symlink_path = os.path.join(out_dir, "openrouter.py")
if os.path.exists(symlink_path) or os.path.islink(symlink_path):
    os.remove(symlink_path)
os.symlink("../openrouter.py", symlink_path)

# MODEL.md
params_list = "\n".join(f"- `{p}`" for p in supported_params)
with open(os.path.join(out_dir, "MODEL.md"), "w") as f:
    f.write(f"""# {name}

## Model ID
`{model_id}`

## Description
{description}

## Capabilities
- **Modality**: {modality}
- **Input**: {", ".join(input_modalities)}
- **Context**: {context_length:,} tokens
- **Max completion**: {max_completion:,} tokens

## Pricing (per million tokens)
| Direction | Per token | Per M tokens |
|-----------|-----------|--------------|
| Input     | ${prompt_price} | ${prompt_per_m:.4f} |
| Output    | ${completion_price} | ${completion_per_m:.4f} |

## Supported Parameters
{params_list}
""")

# --- Summary ---
print(f"Overlaid OpenRouter files onto: {out_dir}/")
print(f"  Model:        {model_id}")
print(f"  Name:         {name}")
print(f"  Modality:     {modality}")
print(f"  Context:      {context_length:,}")
print(f"  Input $/M:    ${prompt_per_m:.4f}")
print(f"  Output $/M:   ${completion_per_m:.4f}")
print(f"  Capabilities: {capabilities}")
print()
print("Overwritten files:")
print(f"  inference.py, inf.yml, __init__.py, requirements.txt")
print("Added files:")
print(f"  openrouter.py -> ../openrouter.py")
print(f"  MODEL.md")
print()
print("Next: review inf.yml description/images, then deploy:")
print(f"  cd {app_dir} && belt app deploy")
PYSCRIPT
