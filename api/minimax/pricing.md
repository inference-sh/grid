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

## Image Generation (H3 Image)

Billed per token. Source: https://platform.minimax.io/docs/pricing/overview

| Model | Input text /M | Input image /M | Output image /M |
|-------|---------------|----------------|-----------------|
| **MiniMax-H3-Image** | $2.30 | $1.50 | $3.80 |
| MiniMax-H3-Image-Highspeed (no app yet) | $2.30 | $3.00 | $7.60 |

- Reference images bill as input image tokens; there is no per-image fee.
- Prompt enhancement bills as input text tokens (about 2,300 to 3,200 per image; 8 with enhancement off).
- Only images that were generated are billed.
- Vendor promotion until 2026-11-17 16:00 UTC: 50% off the request when the image is at least 16,600,000 px (4K at 1:1 is 16,777,216). The app prices at list rates.

### Measured on minimax/h3-image (2026-11-10)

| Run | Size | Text in | Image in | Image out | List cost | Microcents |
|-----|------|---------|----------|-----------|-----------|------------|
| 1K 16:9 | 1360x768 | 2,479 | 0 | 4,080 | $0.021206 | 2,120,570 |
| 2K auto | 2048x2048 | 2,472 | 0 | 4,096 | $0.021250 | 2,125,040 |
| 4K 1:1 | 4096x4096 | 2,941 | 0 | 16,384 | $0.069024 | 6,902,350 |
| 1K 1:1, enhancement off | 1024x1024 | 8 | 0 | 4,096 | $0.015583 | 1,558,320 |
| 2 images, 1024x768 | 1024x768 | 4,691 | 0 | 8,140 | $0.041721 | 4,172,130 |
| 1K edit, 1 reference (1360x768) | 1360x768 | 3,140 | 3,650 | 4,080 | $0.028201 | 2,820,100 |

Output image tokens: 1K is pixels/256 (at most 4,096), 2K is 4,096, 4K is pixels/1024 (16,384 at 1:1).

### OutputMeta (h3-image)
- `outputs[0]` — ImageMeta: `width`, `height` of the first image, `count` = images generated
- `outputs[0].extra.input_text_tokens`, `input_image_tokens`, `output_image_tokens` — billed tokens for the whole request
- `outputs[0].extra.generated_images`, `requested_images`, `input_image_count`, `total_pixels`, `max_pixels`, `resolution` ("1K", "2K", "4K" or "custom"), `mode` ("text-to-image" or "image-edit")
- `inputs[]` — one ImageMeta per reference image, with `width`, `height`

Microcents per token: text in 230, image in 150, image out 380.

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

## Speech (TTS)

| Model | Per M Characters |
|-------|-----------------|
| **speech-2.8-hd** | $100/M chars ($0.0001/char) |
| **speech-2.8-turbo** | $60/M chars ($0.00006/char) |

## Music

| Model | Per Track |
|-------|----------|
| **music-3.0** | $0.15 per track (up to 5 min) |
| **music-cover** | $0.15 per cover |

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
