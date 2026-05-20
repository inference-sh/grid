# Anthropic Claude Pricing

All prices per million tokens (MTok). Source: https://platform.claude.com/docs/en/about-claude/models/overview

## Standard API Pricing

| Model | Input | Output |
|-------|-------|--------|
| Claude Opus 4.7 | $5.00 | $25.00 |
| Claude Opus 4.6 | $5.00 | $25.00 |
| Claude Opus 4.5 | $5.00 | $25.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Sonnet 4.5 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |

## Prompt Caching Pricing

| Model | 5m Cache Write (1.25x) | 1h Cache Write (2x) | Cache Hit (0.1x) |
|-------|----------------------|---------------------|-------------------|
| Opus 4.7 | $6.25 | $10.00 | $0.50 |
| Opus 4.6 | $6.25 | $10.00 | $0.50 |
| Opus 4.5 | $6.25 | $10.00 | $0.50 |
| Sonnet 4.6 | $3.75 | $6.00 | $0.30 |
| Sonnet 4.5 | $3.75 | $6.00 | $0.30 |
| Haiku 4.5 | $1.25 | $2.00 | $0.10 |

## Batch API Pricing (50% discount)

| Model | Batch Input | Batch Output |
|-------|------------|--------------|
| Opus 4.7 | $2.50 | $12.50 |
| Opus 4.6 | $2.50 | $12.50 |
| Opus 4.5 | $2.50 | $12.50 |
| Sonnet 4.6 | $1.50 | $7.50 |
| Sonnet 4.5 | $1.50 | $7.50 |
| Haiku 4.5 | $0.50 | $2.50 |

## CEL Pricing Reference (for pricing agent)

### partner_expression variables needed

All apps use token-based pricing. The `output_meta` reports `TextMeta(tokens=N)` for both inputs and outputs.

```
# Opus 4.7 / 4.6 / 4.5: $5/MTok in, $25/MTok out
partner_input_per_million:  500000000   # $5.00 in microcents
partner_output_per_million: 2500000000  # $25.00 in microcents

# Sonnet 4.6 / 4.5: $3/MTok in, $15/MTok out
partner_input_per_million:  300000000   # $3.00 in microcents
partner_output_per_million: 1500000000  # $15.00 in microcents

# Haiku 4.5: $1/MTok in, $5/MTok out
partner_input_per_million:  100000000   # $1.00 in microcents
partner_output_per_million: 500000000   # $5.00 in microcents
```

### Example CEL expressions

```cel
# partner_expression
double(text_tokens(inputs)) / 1000000.0 * double(prices.partner_input_per_million) +
double(text_tokens(outputs)) / 1000000.0 * double(prices.partner_output_per_million)

# total_expression
partner_fee
```

## Additional Costs

- **Web Search**: $10 per 1,000 searches (not currently exposed)
- **Web Fetch**: No additional charge
- **US-only inference**: 1.1x multiplier (`inference_geo: "us"`)
- **Fast Mode** (Opus 4.6/4.7 only): 6x standard ($30/$150 per MTok) — not used in our apps
- **Extended output** (Batch API only): Up to 300k tokens via `output-300k-2026-03-24` beta header

## Deprecation Notes

- Claude Opus 4.5 (`claude-opus-4-5-20251101`) — legacy, may be deprecated
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) — legacy, may be deprecated
- Claude Sonnet 4 and Opus 4 — deprecated, retiring June 15, 2026
