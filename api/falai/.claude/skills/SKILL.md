---
name: falai-skills
description: Skills for building fal.ai model wrappers on inference.sh. Use when implementing fal.ai models, searching fal endpoints, configuring pricing, or building inference.sh apps that use fal.ai.
---

# fal.ai Skills

Skills for building inference.sh apps that wrap fal.ai model endpoints.

## Authentication

**fal.ai API Key:** `../.fal.key`

Load it before using fal.ai API:
```bash
export FAL_KEY=$(cat ../.fal.key)
```

## Workflow: Adding a New fal.ai Model

Use these skills in order:

### 1. Search for the model
**Skill:** [fal-model-search](fal-model-search/)

Find the fal.ai endpoint and get its OpenAPI schema:
```bash
curl "https://api.fal.ai/v1/models?endpoint_id=fal-ai/MODEL_NAME&expand=openapi-3.0"
```

### 2. Check if already implemented
**Skill:** [fal-implemented-models](fal-implemented-models/)

Check `../IMPLEMENTED_MODELS.md` to see if model exists.

### 3. Add the model
**Skill:** [fal-add-model](fal-add-model/)

**IMPORTANT:** This skill has a strict checklist. Follow each step IN ORDER:

1. Fetch OpenAPI schema
2. Write MODEL.md - **STOP until complete**
3. Fetch pricing
4. Write PRICING.md - **STOP until complete**
5. Run `infsh app init`
6. Implement inference.py
7. Test with `infsh run`
8. Update IMPLEMENTED_MODELS.md
9. Deploy with `infsh deploy`

### 4. Configure pricing
**Skill:** [fal-pricing](fal-pricing/)

Fetch pricing and create CEL expressions for billing.

## General inference.sh Skills

**Skill:** [creating-apps](creating-apps/)

For general inference.sh development not specific to fal.ai:

| Skill | Purpose |
|-------|---------|
| [using-the-cli](creating-apps/using-the-cli/) | CLI commands |
| [writing-app-logic](creating-apps/writing-app-logic/) | inference.py patterns |
| [configuring-resources](creating-apps/configuring-resources/) | inf.yml setup |
| [managing-secrets](creating-apps/managing-secrets/) | API keys and secrets |
| [tracking-usage](creating-apps/tracking-usage/) | Output metadata for billing |
| [handling-cancellation](creating-apps/handling-cancellation/) | Long-running tasks |
| [optimizing-performance](creating-apps/optimizing-performance/) | Best practices |
| [debugging-issues](creating-apps/debugging-issues/) | Troubleshooting |

## Quick Reference

| Task | Skill |
|------|-------|
| "Add fal.ai model X" | fal-add-model |
| "Find fal.ai models for Y" | fal-model-search |
| "Is model X implemented?" | fal-implemented-models |
| "Set up pricing for X" | fal-pricing |
| "How do I write inference.py?" | creating-apps/writing-app-logic |
| "Configure GPU resources" | creating-apps/configuring-resources |

## Directory Structure

```
../
├── .fal.key                    # fal.ai API key
├── IMPLEMENTED_MODELS.md       # Tracking file
├── fal_helper.py               # Shared helper (symlinked into each app)
├── _skills/                    # These skills
│   ├── SKILL.md                # This file
│   ├── fal-add-model/
│   ├── fal-model-search/
│   ├── fal-implemented-models/
│   ├── fal-pricing/
│   └── creating-apps/
└── [app-name]/                 # Each implemented app
    ├── inference.py
    ├── inf.yml
    ├── requirements.txt
    ├── __init__.py
    ├── fal_helper.py
    ├── MODEL.md                # Schema documentation
    └── PRICING.md              # Pricing documentation
```
