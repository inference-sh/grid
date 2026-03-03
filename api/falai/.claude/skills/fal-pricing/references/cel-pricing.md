# CEL Pricing Reference

Complete reference for CEL (Common Expression Language) pricing expressions.

## Table of Contents

1. [Context Variables](#context-variables)
2. [Output Metadata Types](#output-metadata-types)
3. [Pricing Patterns by Category](#pricing-patterns-by-category)
4. [Description Templates](#description-templates)
5. [Advanced Patterns](#advanced-patterns)

## Context Variables

### Available in All Expressions

| Variable | Type | Description |
|----------|------|-------------|
| `inputs` | Array | Input metadata (e.g., `{type: "text", tokens: 500}`) |
| `outputs` | Array | Output metadata (e.g., `{type: "image", width: 1024, height: 1024}`) |
| `prices` | Map | Price variables defined in form (values in microcents) |
| `resource_cost` | int | Base compute cost in microcents |
| `resource_ms` | int | Execution time in milliseconds |

### Available in total_expression Only

| Variable | Type | Description |
|----------|------|-------------|
| `inference_fee` | int | Result of inference_expression |
| `royalty_fee` | int | Result of royalty_expression |
| `partner_fee` | int | Result of partner_expression |

## Output Metadata Types

### TextMeta (type: "text")

For LLM token-based inputs/outputs.

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | int | Token count. In inputs[] = input tokens, outputs[] = output tokens |
| `extra` | dict | Optional app-specific key-value pairs |

**Example output:**
```json
{"type": "text", "tokens": 1500}
```

### ImageMeta (type: "image")

For image generation/processing.

| Field | Type | Description |
|-------|------|-------------|
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `resolution_mp` | float | Resolution in megapixels (width * height / 1_000_000) |
| `steps` | int | Number of diffusion steps |
| `count` | int | Number of images generated |
| `extra` | dict | Optional app-specific key-value pairs |

**Example output:**
```json
{"type": "image", "width": 1024, "height": 1024, "resolution_mp": 1.048576, "steps": 30, "count": 1}
```

### VideoMeta (type: "video")

For video generation/processing.

| Field | Type | Description |
|-------|------|-------------|
| `width` | int | Video width in pixels |
| `height` | int | Video height in pixels |
| `resolution_mp` | float | Resolution in megapixels per frame |
| `resolution` | string | Standard preset: "480p", "720p", "1080p", "1440p", "4k" |
| `seconds` | float | Duration in seconds |
| `fps` | int | Frames per second |
| `extra` | dict | Optional (e.g., `{generate_audio: true}`) |

**Example output:**
```json
{"type": "video", "width": 1280, "height": 720, "resolution": "720p", "seconds": 5.0, "fps": 24}
```

### AudioMeta (type: "audio")

For audio generation/processing.

| Field | Type | Description |
|-------|------|-------------|
| `seconds` | float | Duration in seconds |
| `sample_rate` | int | Sample rate in Hz |
| `extra` | dict | Optional app-specific key-value pairs |

**Example output:**
```json
{"type": "audio", "seconds": 30.0, "sample_rate": 44100}
```

### RawMeta (type: "raw")

For custom pricing with direct cost specification.

| Field | Type | Description |
|-------|------|-------------|
| `cost` | float | Cost in dollar cents |
| `extra` | dict | Optional app-specific key-value pairs |

**Example output:**
```json
{"type": "raw", "cost": 0.5}
```

## Pricing Patterns by Category

### Image Generation

**Megapixel-based:**
```cel
(double(outputs[0].width) * double(outputs[0].height) / 1000000.0) * double(prices.per_megapixel)
```

**Per-image flat rate:**
```cel
double(outputs[0].count) * double(prices.per_image)
```

**Step-based (more steps = higher cost):**
```cel
(double(outputs[0].width) * double(outputs[0].height) / 1000000.0) * (double(outputs[0].steps) / 30.0) * double(prices.per_megapixel)
```

### Video Generation

**Token-based (width × height × fps × seconds):**
```cel
(double(outputs[0].width) * double(outputs[0].height) * double(outputs[0].fps) * double(outputs[0].seconds) / 1000000.0) * double(prices.per_million_tokens)
```

**Per-second duration:**
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

**Resolution + duration:**
```cel
double(outputs[0].resolution_mp) * double(outputs[0].seconds) * double(prices.per_mp_second)
```

### Audio Generation

**Per-second:**
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

**Per-minute:**
```cel
(double(outputs[0].seconds) / 60.0) * double(prices.per_minute)
```

### LLM / Text

**Per million output tokens:**
```cel
(double(outputs[0].tokens) / 1000000.0) * double(prices.per_million_output)
```

**Input + output tokens:**
```cel
(double(inputs[0].tokens) / 1000000.0) * double(prices.per_million_input) + (double(outputs[0].tokens) / 1000000.0) * double(prices.per_million_output)
```

### Flat Fee

**Per run:**
```cel
double(prices.per_run)
```

### Raw Cost Passthrough

When app reports exact cost:
```cel
double(outputs[0].cost) * 1000000.0
```

## Description Templates

Human-readable pricing for display.

### Per Megapixel
```cel
"$" + string(double(prices.per_megapixel) / 100000000.0) + " per megapixel"
```

### Per Second
```cel
"$" + string(double(prices.per_second) / 100000000.0) + " per second"
```

### Per Million Tokens
```cel
"$" + string(double(prices.per_million_tokens) / 100000000.0) + " per million tokens"
```

### Flat Fee
```cel
"$" + string(double(prices.per_run) / 100000000.0) + " per run"
```

### Combined (Input + Output)
```cel
"$" + string(double(prices.per_million_input) / 100000000.0) + "/M input, $" + string(double(prices.per_million_output) / 100000000.0) + "/M output"
```

## Advanced Patterns

### Conditional Pricing (Extra Field)

When app uses extra field for variants:
```cel
outputs[0].extra.generate_audio == true
  ? double(outputs[0].seconds) * double(prices.with_audio)
  : double(outputs[0].seconds) * double(prices.video_only)
```

### Minimum Price

Ensure minimum charge:
```cel
max(double(prices.minimum), (double(outputs[0].width) * double(outputs[0].height) / 1000000.0) * double(prices.per_megapixel))
```

### Tiered Pricing

Lower rate after threshold:
```cel
outputs[0].seconds <= 10.0
  ? double(outputs[0].seconds) * double(prices.tier1)
  : 10.0 * double(prices.tier1) + (double(outputs[0].seconds) - 10.0) * double(prices.tier2)
```

### Multiple Outputs

Sum cost across all outputs:
```cel
outputs.map(o, double(o.width) * double(o.height) / 1000000.0 * double(prices.per_megapixel)).sum()
```

## Price Variable Naming Conventions

| Variable Name | Use Case |
|---------------|----------|
| `per_megapixel` | Image generation |
| `per_image` | Flat per-image |
| `per_million_tokens` | Video tokens |
| `per_second` | Audio/video duration |
| `per_minute` | Longer audio |
| `per_million_input` | LLM input tokens |
| `per_million_output` | LLM output tokens |
| `per_run` | Flat fee |
| `minimum` | Minimum charge |

## Converting fal.ai Prices

fal.ai returns prices in USD dollars. Convert to microcents:

```
microcents = dollars * 100000000
```

Example: `$0.025` → `2500000` microcents
