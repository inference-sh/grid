# Anthropic Claude Pricing

All prices per million tokens (MTok). Source: https://platform.claude.com/docs/en/about-claude/models/overview

## Standard API Pricing

| Model | Input | Output |
|-------|-------|--------|
| Claude Fable 5 | $10.00 | $50.00 |
| Claude Mythos 5 | $10.00 | $50.00 |
| Claude Opus 5 | $5.00 | $25.00 |
| Claude Opus 4.8 | $5.00 | $25.00 |
| Claude Opus 4.7 | $5.00 | $25.00 |
| Claude Opus 4.6 | $5.00 | $25.00 |
| Claude Opus 4.5 | $5.00 | $25.00 |
| Claude Sonnet 5 (through 2026-08-31) | $2.00 | $10.00 |
| Claude Sonnet 5 (from 2026-09-01) | $3.00 | $15.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Sonnet 4.5 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |

## Prompt Caching Pricing

| Model | 5m Cache Write (1.25x) | 1h Cache Write (2x) | Cache Hit (0.1x) |
|-------|----------------------|---------------------|-------------------|
| Fable 5 / Mythos 5 | $12.50 | $20.00 | $1.00 |
| Opus 5 | $6.25 | $10.00 | $0.50 |
| Opus 4.8 | $6.25 | $10.00 | $0.50 |
| Opus 4.7 | $6.25 | $10.00 | $0.50 |
| Opus 4.6 | $6.25 | $10.00 | $0.50 |
| Opus 4.5 | $6.25 | $10.00 | $0.50 |
| Sonnet 5 (through 2026-08-31) | $2.50 | $4.00 | $0.20 |
| Sonnet 4.6 | $3.75 | $6.00 | $0.30 |
| Sonnet 4.5 | $3.75 | $6.00 | $0.30 |
| Haiku 4.5 | $1.25 | $2.00 | $0.10 |

## Batch API Pricing (50% discount)

| Model | Batch Input | Batch Output |
|-------|------------|--------------|
| Fable 5 / Mythos 5 | $5.00 | $25.00 |
| Opus 5 | $2.50 | $12.50 |
| Opus 4.8 | $2.50 | $12.50 |
| Opus 4.7 | $2.50 | $12.50 |
| Opus 4.6 | $2.50 | $12.50 |
| Opus 4.5 | $2.50 | $12.50 |
| Sonnet 5 (through 2026-08-31) | $1.00 | $5.00 |
| Sonnet 4.6 | $1.50 | $7.50 |
| Sonnet 4.5 | $1.50 | $7.50 |
| Haiku 4.5 | $0.50 | $2.50 |

## CEL Pricing Reference (for pricing agent)

### partner_expression variables needed

All apps use token-based pricing. The `output_meta` reports `TextMeta(tokens=N)` for both inputs and outputs.

**Units: price variables are microcents, where $1 = 100,000,000 microcents.** So $1/MTok is
`100000000` and $5/MTok is `500000000`. The pricing agent has repeatedly saved these 100x too
low (`5000000`, i.e. $0.05/MTok) even when given the dollar rate, so verify before approving a
publish: for a 41-input / 282-output-token task at $5/$25, `evaluate` must return a total of
`725500` microcents ($0.007255). A total of `7255` means the variables are 100x too low.

```
# Fable 5 / Mythos 5: $10/MTok in, $50/MTok out
partner_input_per_million:  1000000000  # $10.00 in microcents
partner_output_per_million: 5000000000  # $50.00 in microcents

# Opus 5 / 4.8 / 4.7 / 4.6 / 4.5: $5/MTok in, $25/MTok out
partner_input_per_million:  500000000   # $5.00 in microcents
partner_output_per_million: 2500000000  # $25.00 in microcents

# Sonnet 5 (intro, through 2026-08-31): $2/MTok in, $10/MTok out
partner_input_per_million:  200000000   # $2.00 in microcents
partner_output_per_million: 1000000000  # $10.00 in microcents

# Sonnet 4.6 / 4.5 (and Sonnet 5 from 2026-09-01): $3/MTok in, $15/MTok out
partner_input_per_million:  300000000   # $3.00 in microcents
partner_output_per_million: 1500000000  # $15.00 in microcents

# Haiku 4.5: $1/MTok in, $5/MTok out
partner_input_per_million:  100000000   # $1.00 in microcents
partner_output_per_million: 500000000   # $5.00 in microcents
```

### Example CEL expressions

```cel
# partner_expression (cache-aware)
# input_tokens from Anthropic = uncached tokens only
# cache_read_tokens = tokens read from cache (billed at 0.1x input price)
# cache_write_tokens = tokens written to cache (billed at 1.25x input price)
# total tokens in TextMeta = uncached + cache_read + cache_write
(
  double(get_extra(inputs[0], "cache_read_tokens", 0)) / 1000000.0 * double(prices.partner_cache_read_per_million) +
  double(get_extra(inputs[0], "cache_write_tokens", 0)) / 1000000.0 * double(prices.partner_cache_write_per_million) +
  double(text_tokens(inputs) - get_extra(inputs[0], "cache_read_tokens", 0) - get_extra(inputs[0], "cache_write_tokens", 0)) / 1000000.0 * double(prices.partner_input_per_million)
) +
double(text_tokens(outputs)) / 1000000.0 * double(prices.partner_output_per_million)

# total_expression
partner_fee
```

### Cache pricing constants

Cache read = 0.1x input price. Cache write (5min) = 1.25x input price.

```
# Fable 5 / Mythos 5: input=$10
partner_cache_read_per_million:   100000000   # $1.00
partner_cache_write_per_million: 1250000000   # $12.50

# Opus 5 / 4.8 / 4.7 / 4.6 / 4.5: input=$5
partner_cache_read_per_million:    50000000   # $0.50
partner_cache_write_per_million:  625000000   # $6.25

# Sonnet 5 (intro): input=$2
partner_cache_read_per_million:    20000000   # $0.20
partner_cache_write_per_million:  250000000   # $2.50

# Sonnet 4.6 / 4.5: input=$3
partner_cache_read_per_million:    30000000   # $0.30
partner_cache_write_per_million:  375000000   # $3.75

# Haiku 4.5: input=$1
partner_cache_read_per_million:    10000000   # $0.10
partner_cache_write_per_million:  125000000   # $1.25
```

## Additional Costs

- **Web Search**: $10 per 1,000 searches (not currently exposed)
- **Web Fetch**: No additional charge
- **US-only inference**: 1.1x multiplier (`inference_geo: "us"`)
- **Fast Mode** (Opus 5 / Opus 4.8 only, Claude API only): $10/$50 per MTok — not used in our apps. Removed on Opus 4.7 (`speed: "fast"` errors); no-op on Opus 4.6.
- **Extended output** (Batch API only): Up to 300k tokens via `output-300k-2026-03-24` beta header

## Model Notes

- **Claude Fable 5** (`claude-fable-5`) — GA, adaptive thinking (always on), 1M context, 128k output
- **Claude Mythos 5** (`claude-mythos-5`) — limited availability (Project Glasswing), adaptive thinking (always on), 1M context, 128k output

## Deprecation Notes

- Claude Opus 4.5 (`claude-opus-4-5-20251101`) — legacy, may be deprecated
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) — legacy, may be deprecated
- Claude Sonnet 4 and Opus 4 — deprecated, retiring June 15, 2026
