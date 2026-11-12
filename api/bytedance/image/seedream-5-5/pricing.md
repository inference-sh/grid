# Seedream 5.5 pricing

Model: `dola-seedream-5-5-261028` (BytePlus ModelArk, ap-southeast-1).
Source: https://ai.byteplus.com/ark/region:ap-southeast-1/docs/model-pricing (Image generation models table), read 2026-11-12.

## Vendor rates

| Item | Vendor price | Microcents |
| --- | --- | --- |
| Input (reference) image, each, from the first | $0.008 | 800,000 |
| Output image, 921,600 <= P <= 1536² × 1.05² (2,601,123.84) px | $0.020 | 2,000,000 |
| Output image, 2,601,123.84 < P <= 2048² × 1.05² (4,624,220.16) px | $0.035 | 3,500,000 |

P is the total pixel count of one output image. Images that fail content moderation are not billed.
Layer decomposition (billed per layer) is not exposed by this app.

Unlike Seedream 5.0 pro (first input image free, then $0.003), every input image is billed.

## Size tiers in the app

The app sends the official WxH for each `size` + `aspect_ratio`. Every 1K and 1.5K size falls in the
$0.020 band (largest is 1792x1344 = 2,408,448 px); every 2K size falls in the $0.035 band
(smallest is 3136x1344 = 4,214,784 px).

| size | 1:1 | 4:3 | 3:4 | 16:9 | 9:16 | 3:2 | 2:3 | 21:9 | Output price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1K | 1024x1024 | 1152x864 | 864x1152 | 1424x800 | 800x1424 | 1248x832 | 832x1248 | 1568x672 | $0.020 |
| 1.5K | 1536x1536 | 1792x1344 | 1344x1792 | 2048x1152 | 1152x2048 | 1872x1248 | 1248x1872 | 2352x1008 | $0.020 |
| 2K | 2048x2048 | 2368x1776 | 1776x2368 | 2816x1584 | 1584x2816 | 2496x1664 | 1664x2496 | 3136x1344 | $0.035 |

## What the app reports (output_meta)

- `inputs`: one `ImageMeta` per reference image (width/height when read locally).
- `outputs[0]`: `ImageMeta(width, height, count=1)` from the response `data[0].size`, with
  `extra.pixel_tier` = `low` (P <= 2,601,123.84) or `high`, `extra.size` (`1K`/`1.5K`/`2K`),
  `extra.input_images`, `extra.format`, `extra.background`, `extra.actual_background`.

## Proposed pricing (not published yet)

cost = 800,000 × size(inputs) + (outputs[0].width × outputs[0].height <= 2,601,123.84 ? 2,000,000 : 3,500,000)

Estimate (exact from task_inputs): 800,000 × size(images) + (size == "2K" ? 3,500,000 : 2,000,000).
Defaults (2K, no images) = 3,500,000 µc ($0.035). Ten images at 2K = 11,500,000 µc ($0.115).
