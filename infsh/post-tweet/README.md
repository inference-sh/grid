# Inference.sh App

Build and deploy applications on the inference.sh platform.

## Development Environment

### Prerequisites

**uv** (Python package manager from astral.sh):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Hardware Requirements

| App Type | Development Environment |
|----------|------------------------|
| CPU apps | Any machine |
| GPU apps | Requires NVIDIA CUDA GPU |

> GPU apps need a CUDA-capable GPU for local development and testing.

## Quick Start

```bash
# Install CLI
curl -fsSL https://cli.inference.sh | sh
infsh login

# Create app
infsh app init my-app
cd my-app

# Test locally
infsh app run

# Deploy
infsh app deploy
```

## Project Structure

```
your-app/
├── inference.py      # Main logic (setup, run)
├── inf.yml          # Configuration
├── requirements.txt # Python dependencies (pip)
├── packages.txt     # System dependencies (apt) — optional
├── skills/          # AI coding agent guidance
└── README.md
```

## Core Classes

- **`AppInput`** - Request input schema  
- **`AppOutput`** - Response output schema
- **`App`** - Application with `setup()` and `run()` methods

## Base Images

Apps run in containers:

| Type | Image |
|------|-------|
| GPU | `docker.inference.sh/gpu:latest-cuda` |
| CPU | `docker.inference.sh/cpu:latest` |

> Currently only NVIDIA CUDA GPUs are supported for GPU apps.

## Documentation

Full documentation: **[inference.sh/docs/extend](https://inference.sh/docs/extend)**

| Topic | Link |
|-------|------|
| CLI Setup | [docs/extend/cli-setup](https://inference.sh/docs/extend/cli-setup) |
| Coding Agents | [docs/extend/coding-agents](https://inference.sh/docs/extend/coding-agents) |
| App Code | [docs/extend/app-code](https://inference.sh/docs/extend/app-code) |
| Configuration | [docs/extend/configuration](https://inference.sh/docs/extend/configuration) |
| Secrets | [docs/extend/secrets](https://inference.sh/docs/extend/secrets) |
| Integrations | [docs/extend/integrations](https://inference.sh/docs/extend/integrations) |
| Output Metadata | [docs/extend/output-meta](https://inference.sh/docs/extend/output-meta) |
| Best Practices | [docs/extend/best-practices](https://inference.sh/docs/extend/best-practices) |
| Troubleshooting | [docs/extend/troubleshooting](https://inference.sh/docs/extend/troubleshooting) |

## Examples

Real implementations: [github.com/inference-sh/grid](https://github.com/inference-sh/grid)

## Support

- Docs: [inference.sh/docs](https://inference.sh/docs)
- Discord: [discord.gg/inference](https://discord.gg/inference)
