# Runway Official Pricing

Credit-based system. 1 credit = $0.01.

## Video Generation

| Model | Rate | Notes |
|-------|------|-------|
| **Gen-4.5** (`gen4.5`) | 12 credits/sec ($0.12/sec) | T2V + I2V, 2-10s |
| **Gen-4 Turbo** (`gen4_turbo`) | 5 credits/sec ($0.05/sec) | I2V only |
| **Aleph 2.0** (`aleph2`) | 28 credits/sec ($0.28/sec) | V2V, 56 credit min |
| **Act-Two** (`act_two`) | 5 credits/sec ($0.05/sec) | Character performance |

## Image Generation

| Model | Rate | Notes |
|-------|------|-------|
| **Gen-4 Image** (`gen4_image`) | 5 credits ($0.05) per 720p, 8 credits ($0.08) per 1080p | T2I with reference images |
| **Gen-4 Image Turbo** (`gen4_image_turbo`) | 2 credits ($0.02) per image, any resolution | I2I, requires reference image |

## OutputMeta Mapping

### Video apps (gen-4-5, gen-4-turbo, aleph-2, act-two)
- `outputs[0].seconds` — video duration in seconds
- `outputs[0].width`, `outputs[0].height` — resolution
- `outputs[0].extra.model` — model identifier

### Image apps (gen-4-image, gen-4-image-turbo)
- `outputs[0].width`, `outputs[0].height` — resolution
- `outputs[0].count` — number of images (always 1)

## Microcent Conversion

$1.00 = 100,000,000 microcents

| Per-unit price | Microcents |
|---|---|
| $0.01 (1 credit) | 1,000,000 |
| $0.02 (2 credits) | 2,000,000 |
| $0.05 (5 credits) | 5,000,000 |
| $0.08 (8 credits) | 8,000,000 |
| $0.12/sec (gen4.5) | 12,000,000 |
| $0.28/sec (aleph2) | 28,000,000 |

## CEL Expression Templates

### Per-second video pricing
```cel
# partner_expression
double(outputs[0].seconds) * double(prices.partner_per_second)

# total_expression
partner_fee
```

### Per-image pricing (fixed rate)
```cel
# partner_expression (gen4_image_turbo — flat rate)
double(prices.partner_per_image)

# total_expression
partner_fee
```

### Per-image pricing (resolution-dependent, gen4_image)
```cel
# partner_expression
(outputs[0].height > 720 || outputs[0].width > 1280)
  ? double(prices.partner_per_image_1080p)
  : double(prices.partner_per_image_720p)

# total_expression
partner_fee
```
