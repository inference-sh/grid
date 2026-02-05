---
name: fal-pricing
description: Fetch fal.ai model pricing and create CEL expressions for inference.sh pricing configuration. Use when setting up pricing for a fal.ai app, creating CEL expressions, or configuring pricing descriptions. Triggers on "fal pricing", "fal model cost", "cel pricing expression", "pricing configuration", "set up pricing for fal app".
---

# Fal Pricing

Fetch fal.ai pricing and create PRICING.md with CEL expressions.

## Step 1: Fetch fal.ai Pricing

```bash
curl -s -H "Authorization: Key $FAL_KEY" \
  "https://api.fal.ai/v1/models/pricing?endpoint_id=fal-ai/MODEL_NAME"
```

Response:
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

## Step 2: Determine Pricing Model

| fal.ai Unit | Our Model | Price Variable |
|-------------|-----------|----------------|
| `image` | per megapixel | `per_megapixel` |
| `second` | per second | `per_second` |
| `1000 characters` | per 1k chars | `per_1k_characters` |
| `minute` | per minute | `per_minute` |
| `request` | per run | `per_run` |

## Step 3: Calculate Price Variable

Convert fal.ai price to microcents:

```
microcents = dollars * 100_000_000
```

Examples:
- `$0.025` → `2_500_000` microcents
- `$0.04` → `4_000_000` microcents
- `$0.001` → `100_000` microcents

## Step 4: Write PRICING.md

Create this file in the app directory:

```markdown
# Pricing: [app-name]

## fal.ai Base Price
- Endpoint: `fal-ai/model-name`
- Price: $[unit_price] per [unit]
- Currency: USD

## Price Variables (microcents)
- `per_[unit]`: [microcents] (= $[price] * 100000000)

## CEL Expressions

### inference_expression
```cel
[see patterns below]
```

### pricing_description
```cel
"$[price] per [unit]"
```

## Calculation Notes
[How the fal.ai price maps to our model]
```

---

## CEL Expression Patterns

### Image Generation (per megapixel)

fal.ai: `$0.025 per image` at ~1MP base

```cel
(double(outputs[0].width) * double(outputs[0].height) / 1000000.0) * double(prices.per_megapixel)
```

Description:
```cel
"$" + string(double(prices.per_megapixel) / 100000000.0) + " per megapixel"
```

### Video Generation (per second)

fal.ai: `$0.05 per second`

```cel
double(outputs[0].seconds) * double(prices.per_second)
```

Description:
```cel
"$" + string(double(prices.per_second) / 100000000.0) + " per second"
```

### Audio/TTS (per second)

fal.ai: `$0.001 per second`

```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### Audio/TTS (per 1000 characters)

fal.ai: `$0.04 per 1000 characters`

For TTS where we don't know output duration upfront, use input character count:

```cel
(double(size(inputs[0].text)) / 1000.0) * double(prices.per_1k_characters)
```

Or use RawMeta to pass through exact cost from fal.ai response.

### Flat Fee (per run)

fal.ai: `$0.10 per request`

```cel
double(prices.per_run)
```

Description:
```cel
"$" + string(double(prices.per_run) / 100000000.0) + " per run"
```

### Raw Cost Passthrough

When app calculates exact cost:

```cel
double(outputs[0].cost) * 1000000.0
```

---

## Output Metadata Quick Reference

| Type | Key Fields |
|------|------------|
| `image` | `width`, `height`, `resolution_mp`, `steps`, `count` |
| `video` | `width`, `height`, `seconds`, `fps` |
| `audio` | `seconds`, `sample_rate` |
| `text` | `tokens` |
| `raw` | `cost` (in cents) |

Full reference: `references/cel-pricing.md`

---

## Example: TTS App

fal.ai returns: `$0.04 per 1000 characters`

**PRICING.md:**
```markdown
# Pricing: my-tts-app

## fal.ai Base Price
- Endpoint: `fal-ai/my-tts`
- Price: $0.04 per 1000 characters
- Currency: USD

## Price Variables (microcents)
- `per_1k_characters`: 4000000 (= $0.04 * 100000000)

## CEL Expressions

### inference_expression
```cel
(double(size(inputs[0].text)) / 1000.0) * double(prices.per_1k_characters)
```

### pricing_description
```cel
"$0.04 per 1000 characters"
```

## Calculation Notes
fal.ai charges per 1000 characters of input text.
We measure input text length and apply the same rate.
```

---

## Example: Image Generation App

fal.ai returns: `$0.025 per image` (at 1MP)

**PRICING.md:**
```markdown
# Pricing: my-image-app

## fal.ai Base Price
- Endpoint: `fal-ai/my-image-model`
- Price: $0.025 per image (at 1MP)
- Currency: USD

## Price Variables (microcents)
- `per_megapixel`: 2500000 (= $0.025 * 100000000)

## CEL Expressions

### inference_expression
```cel
(double(outputs[0].width) * double(outputs[0].height) / 1000000.0) * double(prices.per_megapixel)
```

### pricing_description
```cel
"$0.025 per megapixel"
```

## Calculation Notes
fal.ai base price is for ~1MP images.
We scale linearly with actual output resolution.
```
