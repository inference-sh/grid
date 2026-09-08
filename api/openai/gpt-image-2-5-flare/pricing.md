# Pricing: gpt-image-2.5-flare

Third-party image model billed on tokens. Token rates match gpt-image-2:

| Type | Price per 1M tokens |
|------|---------------------|
| Text input | $5.00 |
| Cached text input | $1.25 |
| Image input (edit reference images) | $8.00 |
| Cached image input | $2.00 |
| Image output | $30.00 |

Flare and Sunburst consume the same tokens at the same quality and size (measured 2026-09-08, see table).

## Output Token Formula

Same tile formula as gpt-image-2, with two new quality tiers and re-mapped grids:

```
qualityTiles = { low: 16, medium: 24, high: 48, xhigh: 64, max: 96 }

tiles     = qualityTiles[quality]
longEdge  = max(width, height)
shortEdge = min(width, height)
scaledShort = round(tiles * shortEdge / longEdge)

wTiles = tiles      if width >= height else scaledShort
hTiles = scaledShort if width >= height else tiles
totalTiles = wTiles * hTiles

outputTokens = ceil(totalTiles * (2_000_000 + width * height) / 4_000_000) * 1.016
```

Long edge gets the full grid size, short edge scales proportionally. Metered output tokens
run ~1.6% above the pure tile formula at every tier (196 vs 192, 1756 vs 1728, 7024 vs 6912).

Tier mapping versus gpt-image-2: 2.5 `high` costs what gpt-image-2 `medium` cost, 2.5 `max`
costs what gpt-image-2 `high` cost. 2.5 `medium` and `xhigh` are new grids.

## Measured (1024x1024, $30/M output tokens)

| Quality | Tile Grid | Output Tokens (measured) | Price   |
|---------|-----------|--------------------------|---------|
| low     | 16x16     | 196                      | $0.0059 |
| medium  | 24x24     | 439                      | $0.013  |
| high    | 48x48     | 1,756                    | $0.053  |
| xhigh   | 64x64     | 3,122                    | $0.094  |
| max     | 96x96     | 7,024                    | $0.211  |

Text prompt input was 18 tokens for a one-sentence prompt (negligible).

## Edits

Each 1024x1024 reference image cost 1,033 image input tokens ($0.0083) in a measured edit
(1,051 input tokens total, 18 of them text). Reference images are always processed at high
fidelity; `input_fidelity` is not accepted.

## Notes

- `auto` quality lets the model choose; bill by the `quality` reported in `output_meta.extra`
  and the measured `output_tokens` also stored there.
- Supports custom resolutions: max edge 3840px, both edges multiples of 16px, ratio <= 3:1,
  total pixels 655,360–8,294,400
- Transparent backgrounds via `background: transparent` (png/webp only) cost the same tokens
  as opaque
- Partial image streaming (not exposed) adds 100 output tokens per partial
- Batch API is 50% of these rates (not used)
- Source: https://developers.openai.com/api/docs/models/gpt-image-2.5-flare and
  https://developers.openai.com/api/docs/pricing
