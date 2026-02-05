---
name: fal-implemented-models
description: Track and compare implemented fal.ai models against available endpoints. Use when checking if a fal.ai model is already implemented, listing implemented models, or identifying gaps. Triggers on "implemented fal models", "which fal models do we have", "is this fal model implemented", "fal model status", "compare fal models".
---

# Fal Implemented Models

Track implemented fal.ai models and compare against available endpoints.

## Tracking File

**Location:** `../../IMPLEMENTED_MODELS.md`

This file maintains a table of all implemented fal.ai models with mappings from our app names to fal.ai endpoint IDs.

## Usage

### Check if Model is Implemented

1. Read IMPLEMENTED_MODELS.md
2. Search for the fal.ai endpoint ID in the table
3. If found, note our app name

### List All Implemented Models

```bash
# Get current implementations
cat ../../IMPLEMENTED_MODELS.md
```

### Find Gaps

1. Use `fal-model-search` skill to get available models
2. Compare against IMPLEMENTED_MODELS.md
3. Models in fal.ai but not in our list are gaps

### Update Tracking File

When adding a new model:
1. Add entry to the table in IMPLEMENTED_MODELS.md
2. Update the "Last Updated" date
3. Include: our app name, fal endpoint ID, category

## Directory Check

Quick way to see implemented apps:
```bash
ls -d */ | xargs -I {} basename {}
```

Excludes `skills/` directory which contains these skills.

## Consolidation Strategy

fal.ai often has multiple functions as separate endpoints:
- `fal-ai/model/text-to-image`
- `fal-ai/model/image-to-image`

**Our approach:** Consolidate into single apps where practical:
- If input has image -> run image-to-image variant
- If input is text only -> run text-to-image variant
- Language variants (like TTS with multiple languages) -> single app with language dropdown

This keeps our ecosystem clean and reduces redundancy.

## Example Workflow

User asks: "Do we have flux implemented?"

1. Read IMPLEMENTED_MODELS.md
2. Search for "flux" entries
3. Report: Yes, we have flux-2-dev, flux-2-dev-turbo, flux-2-klein-lora, flux-dev-lora

User asks: "What video models are we missing?"

1. Use fal-model-search to list all `category=image-to-video` and `category=text-to-video`
2. Compare against IMPLEMENTED_MODELS.md video entries
3. Report gaps
