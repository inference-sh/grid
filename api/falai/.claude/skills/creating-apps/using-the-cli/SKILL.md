---
name: using-the-cli
description: Use the inference.sh CLI commands. Use when running app commands, agent commands, deploying, pulling, or managing inference.sh apps via command line.
---

# Using the CLI

Command reference for the inference.sh CLI.

## Prerequisites

### uv (Required)

The CLI uses **uv** for Python environment management:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Hardware

| App Type | Development |
|----------|-------------|
| CPU apps | Any machine |
| GPU apps | NVIDIA CUDA GPU required |

## Installation

```bash
curl -fsSL https://cli.inference.sh | sh
infsh login
infsh me  # Check current user
```

## App Commands

### Development

```bash
# Create
infsh app init my-app              # Non-interactive
infsh app init                     # Interactive

# Test locally
infsh app test                     # Test with input.json
infsh app test --input '{"k":"v"}' # Test with inline JSON
infsh app test --input in.json     # Test with input file
infsh app test --save-example      # Generate sample input.json

# Deploy
infsh app deploy                   # Deploy from current directory
infsh app deploy --dry-run         # Validate without deploying
```

### Running Apps (Cloud)

```bash
# Run apps in the cloud
infsh app run user/app --input input.json
infsh app run user/app@version --input '{"prompt": "hello"}'

# Generate sample input for an app
infsh app sample user/app
infsh app sample user/app --save input.json
```

### Managing Apps

```bash
# Your apps
infsh app my                       # List your deployed apps
infsh app my -l                    # Detailed list

# Browse store
infsh app list                     # List available apps
infsh app list --featured          # Featured apps
infsh app list --category image    # Filter by category

# Get app details
infsh app get user/app             # View app info and schemas
infsh app get user/app --json      # Output as JSON

# Pull apps
infsh app pull [id]                # Pull an app
infsh app pull --all               # Pull all apps
infsh app pull --all --force       # Overwrite existing
```

## Integration Commands

```bash
infsh app integrations list        # List available integrations
```

## General Commands

```bash
infsh help                         # Get help
infsh [command] --help             # Command help
infsh version                      # View version
infsh update                       # Update CLI
infsh completion bash              # Shell completions (bash/zsh/fish)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `INFSH_API_KEY` | API key (overrides config file) |
