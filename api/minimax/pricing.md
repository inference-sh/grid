# MiniMax Official Pricing

Pay-as-you-go API pricing. All prices in USD.

## Video Generation (H3)

| Resolution | Per Second |
|------------|-----------|
| **768P** | $0.08/sec |
| **2K** | $0.13/sec |

### Input Materials
- Audio: free
- Images: first 5 free, then $0.04 per additional image
- Video input: billed at output resolution rates

### Regeneration (768P → 2K)
- $0.05/sec of regenerated output

## LLM (M3) — Standard Tier (50% permanent discount applied)

| Context | Input | Output | Cache Read |
|---------|-------|--------|------------|
| ≤512k tokens | $0.30/M | $1.20/M | $0.06/M |
| >512k tokens | $0.60/M | $2.40/M | $0.12/M |

### Priority Tier (1.5x standard)

| Context | Input | Output | Cache Read |
|---------|-------|--------|------------|
| ≤512k tokens | $0.45/M | $1.80/M | $0.09/M |
| >512k tokens | $0.90/M | $3.60/M | $0.18/M |

## Legacy LLMs

| Model | Input/M | Output/M |
|-------|---------|----------|
| M2.7 / M2.7-highspeed | $0.30-0.60 | $1.20-2.40 |
| M2.5 / M2.5-highspeed | $0.30-0.60 | $1.20-2.40 |
| M2.1 / M2.1-highspeed | $0.30-0.60 | $1.20-2.40 |

## OutputMeta Mapping

### Video apps (h3)
- `outputs[0].seconds` — video duration in seconds
- `outputs[0].width`, `outputs[0].height` — resolution
- `outputs[0].resolution` — "768P" or "2K"
- `outputs[0].extra.mode` — generation mode
- `outputs[0].extra.input_image_count` — number of input images
- `inputs[]` — ImageMeta/VideoMeta/AudioMeta per input media

## Microcent Conversion

$1.00 = 100,000,000 microcents

| Per-unit price | Microcents |
|---|---|
| $0.08/sec (768P) | 8,000,000 |
| $0.13/sec (2K) | 13,000,000 |
| $0.04/image (extra) | 4,000,000 |

## CEL Expression Templates

### Per-second video pricing (resolution-dependent)
```cel
# partner_expression
outputs[0].resolution == "2K"
  ? double(outputs[0].seconds) * double(prices.partner_per_second_2k)
  : double(outputs[0].seconds) * double(prices.partner_per_second_768p)

# input image surcharge (first 5 free)
# + max(0, int(outputs[0].extra.input_image_count) - 5) * double(prices.partner_per_extra_image)

# total_expression
partner_fee
```
