# Pricing: gpt-image-2.6-sunburst

Billed on the tokens OpenAI reports for each run, read from `output_meta.outputs[0].extra`.

| Type | Price per 1M tokens | Microcents per token | Field |
|------|---------------------|----------------------|-------|
| Image output | $30.00 | 3,000 | `output_tokens` |
| Image input (edits, markup) | $8.00 | 800 | `input_image_tokens` |
| Text input | $5.00 | 500 | `input_text_tokens` |
| SVG author input (`gpt-6-luna`) | $0.10 | 10 | `svg_input_tokens` |
| SVG author output (`gpt-6-luna`) | $0.50 | 50 | `svg_output_tokens`, includes reasoning |

The 2.5 tile formula does not apply: `low` and `medium` use far more tokens than on 2.5, and
tokens do not rise with pixels. Bill the measured tokens; use the table for estimates only.

## Measured output tokens (2026-11-05)

Flare and Sunburst matched at every point measured on both. Non-gif runs are one prompt.

| Quality | 1024x1024 | Other sizes |
|---------|-----------|-------------|
| low | 824 | 664 at 1536x1024, 1,668 at 2048x2048 |
| medium | 1,449 | |
| high | 1,758 | 1,413 at 2048x1152 |
| xhigh | 3,122 | |
| max | 7,024 | 13,342 at 3840x2160 |
| auto | 1,758 | depends on the image |

- `n` multiplies: 1,643 tokens for `n=2` at low.
- `complexity` and `background: transparent` do not change tokens.
- GIF, 512x512 frames: 1,172 at low with 9 frames, 1,668 at low with 16 frames, 2,073 at medium
  with 9 frames. The frames render on one sheet, and that sheet must fit OpenAI's pixel budget.
- SVG at low: 824 image tokens plus 4,985 to 7,024 `gpt-6-luna` output tokens (about $0.003).
- Edit: one reference image and one markup layer used 1,102 image input tokens.

## Expected costs (microcents)

| Run | Cost |
|-----|------|
| low, 14 text tokens | 2,479,000 |
| max, 14 text tokens | 21,079,000 |
| markup edit (824 out, 137 text, 1,102 image in) | 3,422,100 |
| svg at low (824 out, 81 text, 1,553 / 4,985 author) | 2,777,280 |

Source: https://developers.openai.com/api/docs/pricing and
https://developers.openai.com/api/docs/models/gpt-image-2.6-sunburst
