---
name: building-inferencesh-apps
description: Build and deploy applications on inference.sh. Use when getting started, understanding the platform, or needing an overview of inference.sh development.
---

# Inference.sh App Development

Build and deploy applications on the inference.sh platform.

## CLI Installation

```bash
curl -fsSL https://cli.inference.sh | sh
```

```bash
infsh update   # Update CLI
infsh login    # Authenticate
infsh me       # Check current user
```

## Quick Start

```bash
infsh app init my-app    # Create app
infsh app run            # Test locally
infsh app deploy         # Deploy
```

## Related Skills

| Skill | Use When |
|-------|----------|
| [using-the-cli](using-the-cli/) | Running CLI commands for apps and agents |
| [writing-app-logic](writing-app-logic/) | Creating inference.py |
| [configuring-resources](configuring-resources/) | Setting up inf.yml |
| [managing-secrets](managing-secrets/) | Handling API keys |
| [using-oauth-integrations](using-oauth-integrations/) | Google Sheets, Drive |
| [tracking-usage](tracking-usage/) | Output metadata for billing |
| [handling-cancellation](handling-cancellation/) | Long-running tasks |
| [optimizing-performance](optimizing-performance/) | Best practices |
| [debugging-issues](debugging-issues/) | Troubleshooting |

## Resources

- **Full Docs**: [inference.sh/docs](https://inference.sh/docs)
- **Examples**: [github.com/inference-sh/grid](https://github.com/inference-sh/grid)
