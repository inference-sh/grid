---
name: fal-add-model
description: Create new fal.ai model apps for inference.sh platform. Use when implementing a new fal.ai model, wrapping a fal endpoint, or adding a fal-based app. Triggers on "add fal model", "implement fal endpoint", "create fal app", "wrap fal model", "new fal integration".
---

# Fal Add Model

Create inference.sh apps that wrap fal.ai model endpoints.

**IMPORTANT:** Follow these steps IN ORDER. Do not skip steps.

## Checklist

```
[ ] Step 1: Fetch model OpenAPI schema
[ ] Step 2: Write MODEL.md with schema details
[ ] Step 3: Fetch pricing data
[ ] Step 4: Write PRICING.md with CEL expressions
[ ] Step 5: Run `infsh app init`
[ ] Step 6: Implement inference.py using MODEL.md as reference
[ ] Step 7: Test with `infsh run`
[ ] Step 8: Update IMPLEMENTED_MODELS.md
[ ] Step 9: Deploy with `infsh deploy`
```

---

## Step 1: Fetch Model OpenAPI Schema

**STOP.** Do this first before anything else.

```bash
curl -s "https://api.fal.ai/v1/models?endpoint_id=fal-ai/MODEL_NAME&expand=openapi-3.0" | jq .
```

For multiple related endpoints (to potentially merge):
```bash
curl -s "https://api.fal.ai/v1/models?endpoint_id=fal-ai/model/text-to-video&endpoint_id=fal-ai/model/image-to-video&expand=openapi-3.0" | jq .
```

Save the response. You need:
- `metadata.description` - Model description
- `metadata.category` - Category (text-to-image, image-to-video, etc.)
- `openapi.components.schemas.*Input` - Input schema
- `openapi.components.schemas.*Output` - Output schema

---

## Step 2: Write MODEL.md

**STOP.** Do not proceed to implementation until MODEL.md exists.

Create `MODEL.md` in a scratchpad or the app directory with this structure:

```markdown
# Model: fal-ai/model-name

## Endpoint
`fal-ai/model-name`

## Category
[from metadata.category]

## Description
[from metadata.description]

## Input Schema

### Required Fields
- `field_name` (type): Description [from schema]

### Optional Fields
- `field_name` (type, default: X): Description [from schema]

## Output Schema
- `output.field` (type): Description

## Notes
- [Any special handling needed]
- [Constraints or limitations]
```

**For multiple endpoints being merged**, create separate files:
- `MODEL_TEXT_TO_VIDEO.md`
- `MODEL_IMAGE_TO_VIDEO.md`

---

## Step 3: Fetch Pricing

```bash
curl -s -H "Authorization: Key $FAL_KEY" \
  "https://api.fal.ai/v1/models/pricing?endpoint_id=fal-ai/MODEL_NAME"
```

Response format:
```json
{
  "prices": [{
    "endpoint_id": "fal-ai/model-name",
    "unit_price": 0.025,
    "unit": "image",
    "currency": "USD"
  }]
}
```

---

## Step 4: Write PRICING.md

**STOP.** Do not proceed until PRICING.md exists.

Use the `fal-pricing` skill for CEL expression help. Create `PRICING.md`:

```markdown
# Pricing: app-name

## fal.ai Base Price
- Endpoint: `fal-ai/model-name`
- Price: $X.XX per [unit]
- Currency: USD

## Price Variables (microcents)
- `per_[unit]`: [value] (calculation: $X.XX * 100000000)

## CEL Expressions

### inference_expression
```cel
[expression from fal-pricing skill patterns]
```

### pricing_description
```cel
"$X.XX per [unit]"
```

## Calculation Notes
[How fal price maps to our pricing model]
```

---

## Step 5: Initialize App

Now you can create the app structure:

```bash
cd /home/ok/inference/grid/api/falai
infsh app init my-app-name
cd my-app-name
cp ../fal_helper.py .
```

Move MODEL.md and PRICING.md into the app directory for reference.

---

## Step 6: Implement

With MODEL.md and PRICING.md as your reference:

### 6a. Update inf.yml

```yaml
name: my-app-name
description: [from MODEL.md description]
category: [video|image|audio|other]

kernel: python-3.11

resources:
  gpu:
    count: 0
    vram: 0
    type: none
  ram: 4000000000

env: {}

secrets:
  - key: FAL_KEY
    description: fal.ai API key for model access
    optional: false
```

### 6b. Update requirements.txt

```txt
pydantic >= 2.0.0
inferencesh
fal-client>=0.4.0
requests>=2.28.0
```

### 6c. Update __init__.py

```python
from .inference import App, AppInput, AppOutput

__all__ = ["App", "AppInput", "AppOutput"]
```

### 6d. Write inference.py

Use `references/template.md` as the code template. Map fields from MODEL.md to your AppInput/AppOutput classes.

Key patterns:
- Use `Field(description="...")` from MODEL.md descriptions
- Use `Enum` classes for constrained values
- Use `Optional[X] = Field(default=None, ...)` for optional fields
- Use `.uri` for file URLs sent to fal.ai (not `.path`)

---

## Step 7: Test

```bash
cd my-app-name
infsh run
# Creates example.input.json
cp example.input.json input.json
# Edit input.json with test values
infsh run input.json
```

---

## Step 8: Update Tracking

Add entry to `../../IMPLEMENTED_MODELS.md`:

```markdown
| my-app-name | fal-ai/model-name | category |
```

Update the "Last Updated" date.

---

## Step 9: Deploy

```bash
infsh deploy
```

---

## Consolidation Patterns

When fal.ai has multiple related endpoints:

| Pattern | Our Approach |
|---------|--------------|
| text-to-X + image-to-X | Single app, detect image input |
| Multiple languages | Single app with language enum |
| Quality variants (pro/turbo) | Single app with quality enum |

**Consolidation code pattern:**
```python
def _get_model_id(self, input_data: AppInput) -> str:
    if input_data.image:
        return "fal-ai/model/image-to-video"
    return "fal-ai/model/text-to-video"
```

---

## Resources

- **Code template:** `references/template.md`
- **Pricing help:** Use `fal-pricing` skill
- **Model search:** Use `fal-model-search` skill
- **fal_helper.py:** `../../fal_helper.py`
