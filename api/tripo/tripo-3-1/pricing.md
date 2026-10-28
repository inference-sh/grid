# Pricing

What Tripo charges for H series `v3.1-20260211`, so the app's price can be set from it.

- source_url: https://developers.tripo3d.ai/en/pricing
- checked: 2026-10-28
- confidence: high. Prepaid API credits, 1 credit = $0.01. Credits can be decimals.

`run` reports `outputs[0].extra.credits` (Tripo's `credits_consumed` for the task). Price it as credits x 1,000,000 microcents.

| mode | no texture | standard/fast | detailed | extreme |
|------|-----------:|--------------:|---------:|--------:|
| text | 10 | 20 | 30 | 40 |
| image | 20 | 30 | 40 | 50 |
| multiview | 20 | 30 | 40 | 50 |

Add-ons: `geometry_quality=detailed` +20, `quad` +5, `smart_low_poly` +10, `generate_parts` +20.
