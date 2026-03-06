# CEL Pricing Reference

Complete reference for CEL (Common Expression Language) pricing expressions.

## Table of Contents

1. [Helper Functions](#helper-functions)
2. [Context Variables](#context-variables)
3. [Output Metadata Types](#output-metadata-types)
4. [Pricing Patterns by Category](#pricing-patterns-by-category)
5. [Description Templates](#description-templates)
6. [Advanced Patterns](#advanced-patterns)

## Helper Functions

Custom functions available in all pricing expressions.

### Conversion Helpers

| Function | Description | Example |
|----------|-------------|---------|
| `to_cents(x)` | Convert microcents to cents (÷1,000,000) | `to_cents(prices.per_image)` → `3.9` |
| `to_dollars(x)` | Convert microcents to dollars (÷100,000,000) | `to_dollars(prices.per_image)` → `0.039` |

### Math Helpers

| Function | Description | Example |
|----------|-------------|---------|
| `ceil(x)` | Ceiling (round up) | `ceil(1.2)` → `2.0` |
| `floor(x)` | Floor (round down) | `floor(1.8)` → `1.0` |
| `min(a, b)` | Minimum of two values | `min(price, 1000000)` |
| `max(a, b)` | Maximum of two values | `max(price, prices.minimum)` |

### Dimension Helpers

| Function | Description | Example |
|----------|-------------|---------|
| `megapixels(w, h)` | Calculate megapixels | `megapixels(1920, 1080)` → `2.0736` |
| `pixels(w, h)` | Calculate total pixels | `pixels(1920, 1080)` → `2073600` |
| `resolution(w, h)` | Get resolution tier | `resolution(1920, 1080)` → `"1080p"` |

Resolution tiers: `"480p"`, `"720p"`, `"1080p"`, `"1440p"`, `"4k"`

### List Helpers

| Function | Description | Example |
|----------|-------------|---------|
| `sum(list)` | Sum numbers in a list | `sum([1, 2, 3])` → `6` |
| `count_type(list, type)` | Count items by type | `count_type(outputs, "image")` → `2` |
| `first(list, type)` | First item of type (or null) | `first(outputs, "video").seconds` |
| `text_tokens(list)` | Sum tokens from text items | `text_tokens(outputs)` → `1500` |
| `image_count(list)` | Count images (respects count field) | `image_count(outputs)` → `4` |
| `video_seconds(list)` | Sum seconds from video items | `video_seconds(outputs)` → `15.5` |

### Safe Access Helpers

| Function | Description | Example |
|----------|-------------|---------|
| `get(map, key, default)` | Get map value with default | `get(task_inputs, "duration", 8)` |
| `get_extra(item, key, default)` | Get extra field with default | `get_extra(outputs[0], "generate_audio", false)` |

### Before/After Examples

```cel
// BEFORE: Ceiling for megapixel rounding
(double(outputs[0].resolution_mp) - double(int(double(outputs[0].resolution_mp))) > 0.0
  ? double(int(double(outputs[0].resolution_mp))) + 1.0
  : double(int(double(outputs[0].resolution_mp)))) * double(prices.per_megapixel)

// AFTER:
ceil(outputs[0].resolution_mp) * prices.per_megapixel
```

```cel
// BEFORE: Safe input access with fallback
has(task_inputs.duration) ? task_inputs.duration : 8

// AFTER: get() safely returns default when key is missing
get(task_inputs, "duration", 8)
```

```cel
// BEFORE: Check extra field with multiple has() calls
has(outputs[0].extra) && has(outputs[0].extra.generate_audio) && outputs[0].extra.generate_audio == true
  ? prices.with_audio : prices.video_only

// AFTER: get_extra() handles all the safety checks
get_extra(outputs[0], "generate_audio", false) ? prices.with_audio : prices.video_only
```

```cel
// BEFORE: Token pricing with safety checks
(size(outputs) > 0 && outputs[0].type == "text"
  ? (double(outputs[0].tokens) / 1000000.0) * double(prices.per_million_output)
  : 0.0)

// AFTER:
text_tokens(outputs) / 1000000.0 * prices.per_million_output
```

```cel
// BEFORE: Description template with conversion
"$" + string(double(prices.per_image) / 100000000.0) + " per image"

// AFTER:
"$" + string(to_dollars(prices.per_image)) + " per image"
```

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
