# Pricing: gpt-image-2

## Output Token Formula

```
qualityTiles = { low: 16, medium: 48, high: 96 }

tiles     = qualityTiles[quality]
longEdge  = max(width, height)
shortEdge = min(width, height)
scaledShort = round(tiles * shortEdge / longEdge)

wTiles = tiles      if width >= height else scaledShort
hTiles = scaledShort if width >= height else tiles
totalTiles = wTiles * hTiles

outputTokens = ceil(totalTiles * (2_000_000 + width * height) / 4_000_000)
```

Long edge gets the full grid size, short edge scales proportionally. The `(2M + pixels) / 4M` multiplier means square images cost more per tile — at 1024x1024 it's ~0.76x tiles, while 1024x1536 is ~0.89x but with fewer total tiles from aspect ratio scaling.

## Price at $30/M output tokens

| Setting          | Tile Grid | Total Tiles | Output Tokens | Price   |
|------------------|-----------|-------------|---------------|---------|
| Low 1024x1024    | 16x16     | 256         | 195           | $0.006  |
| Low 1024x1536    | 11x16     | 176         | 157           | $0.005  |
| Low 1536x1024    | 16x11     | 176         | 157           | $0.005  |
| Medium 1024x1024 | 48x48     | 2,304       | 1,756         | $0.053  |
| Medium 1024x1536 | 32x48     | 1,536       | 1,372         | $0.041  |
| Medium 1536x1024 | 48x32     | 1,536       | 1,372         | $0.041  |
| High 1024x1024   | 96x96     | 9,216       | 7,024         | $0.211  |
| High 1024x1536   | 64x96     | 6,144       | 5,488         | $0.165  |
| High 1536x1024   | 96x64     | 6,144       | 5,488         | $0.165  |

## Notes

- Supports custom resolutions: max edge 3840px, both edges multiples of 16px, ratio <= 3:1, total pixels 655,360–8,294,400
- `auto` quality and size let the model choose based on prompt
- Partial image streaming adds 100 output tokens per partial
- Edit requests with reference images incur additional input image tokens (always high fidelity)
- Transparent backgrounds NOT supported on gpt-image-2
- JPEG output is faster than PNG
