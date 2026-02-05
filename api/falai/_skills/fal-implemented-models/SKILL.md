---
name: fal-implemented-models
description: Track and compare implemented fal.ai models against available endpoints. Use when checking if a fal.ai model is already implemented, listing implemented models, or identifying gaps. Triggers on "implemented fal models", "which fal models do we have", "is this fal model implemented", "fal model status", "compare fal models".
---

# Fal Implemented Models

Track implemented fal.ai models and compare against available endpoints.

## Database File

**Location:** `../../models.json`

JSON database of all fal.ai model endpoints with implementation tracking.

### Schema

```json
{
  "endpoint_id": "fal-ai/flux-1/dev",
  "category": "text-to-image",
  "display_name": "FLUX.1 [dev]",
  "status": "active",
  "implemented": true,
  "direct": false
}
```

- `implemented`: We have an inference.sh app for this endpoint
- `direct`: We integrate with the provider directly (google, xai, bytedance) rather than via fal.ai

## Usage

### Check if Model is Implemented

```bash
jq '.[] | select(.endpoint_id == "fal-ai/flux-1/dev") | .implemented' models.json
```

### List All Implemented Models

```bash
jq '[.[] | select(.implemented)] | .[].endpoint_id' models.json
```

### List Unimplemented Models

```bash
jq '[.[] | select(.implemented == false)] | .[].endpoint_id' models.json
```

### Filter by Category

```bash
# All text-to-image models
jq '[.[] | select(.category == "text-to-image")]' models.json

# Unimplemented video models
jq '[.[] | select((.category | test("video")) and .implemented == false)]' models.json
```

### Direct Integrations

```bash
# List endpoints we integrate directly (not via fal.ai)
jq '[.[] | select(.direct)] | .[].endpoint_id' models.json

# Unimplemented direct integration candidates
jq '[.[] | select(.direct and .implemented == false)]' models.json
```

### Get Stats

```bash
jq '{
  total: length,
  implemented: [.[] | select(.implemented)] | length,
  by_category: (group_by(.category) | map({(.[0].category): length}) | add)
}' models.json
```

### Mark Model as Implemented

```bash
jq '(.[] | select(.endpoint_id == "fal-ai/new-model")).implemented = true' models.json > tmp.json && mv tmp.json models.json
```

### Mark Multiple Models (by prefix)

```bash
jq '[.[] | if .endpoint_id | startswith("fal-ai/wan-pro") then .implemented = true else . end]' models.json > tmp.json && mv tmp.json models.json
```

## Updating the Database

To refresh from fal.ai API:

```bash
# Fetch all pages from API
curl -s "https://api.fal.ai/v1/models?limit=100" > page1.json
# ... continue pagination until has_more is false

# Merge preserving implemented status
python3 << 'EOF'
import json

# Load existing
with open('models.json') as f:
    existing = {m['endpoint_id']: m['implemented'] for m in json.load(f)}

# Load new from API
# ... process API data ...

# Preserve implemented status
for m in new_models:
    m['implemented'] = existing.get(m['endpoint_id'], False)
EOF
```

## Consolidation Strategy

fal.ai often has multiple functions as separate endpoints:
- `fal-ai/model/text-to-image`
- `fal-ai/model/image-to-image`

**Our approach:** Consolidate into single apps where practical:
- If input has image -> run image-to-image variant
- If input is text only -> run text-to-image variant
- Language variants (like TTS with multiple languages) -> single app with language dropdown

This keeps our ecosystem clean and reduces redundancy.

## Example Workflows

**"Do we have flux implemented?"**

```bash
jq '[.[] | select(.endpoint_id | test("flux")) | {endpoint_id, implemented}]' models.json
```

**"What video models are we missing?"**

```bash
jq '[.[] | select((.category | test("video")) and .implemented == false)] | .[].endpoint_id' models.json
```

**"Coverage by category"**

```bash
jq 'group_by(.category) | map({
  category: .[0].category,
  total: length,
  implemented: [.[] | select(.implemented)] | length
}) | sort_by(-.total)' models.json
```
