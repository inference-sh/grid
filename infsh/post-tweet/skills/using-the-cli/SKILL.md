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

## App Commands

```bash
# Create
infsh app init my-app              # Non-interactive
infsh app init                     # Interactive

# Run locally
infsh app run                      # Run with input.json
infsh app run --input '{"k":"v"}'  # Run with JSON string
infsh app run --input-file in.json # Custom input file
infsh app run --save-example       # Generate sample input.json

# Deploy
infsh app deploy                   # Deploy from current directory

# Manage
infsh app ls                       # List apps
infsh app ls -l                    # Detailed list
infsh app pull [id]                # Pull an app
infsh app pull --all               # Pull all apps
infsh app pull --all --force       # Overwrite existing
```

## Agent Commands

```bash
infsh agent init my-agent          # Create agent (non-interactive)
infsh agent init                   # Interactive
infsh agent deploy                 # Deploy agent
infsh agent ls                     # List agents
infsh agent pull [id]              # Pull agent
infsh agent chat [id]              # Chat with agent
```

## Integration Commands

```bash
infsh integrations list            # List available integrations
```

## General

```bash
infsh help                         # Get help
infsh [command] --help             # Command help
infsh version                      # View version
infsh update                       # Update CLI
infsh login                        # Authenticate
infsh me                           # Current user
```

📖 **Full docs**: [inference.sh/docs/extend/cli-setup](https://inference.sh/docs/extend/cli-setup)
