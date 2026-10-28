# Pricing

What Hyper3D charges for Rodin Gen-2.5, so the app's price can be set from it.

- source_url: https://docs.hyper3d.ai/en (tier pages, "Pricing"); https://hyper3d.ai/pricing?lang=en
- checked: 2026-10-28
- confidence: medium. API needs the Business plan; a credit costs $0.58 (monthly) or $0.46 (yearly), $1.50 bought directly. Whether HighPack costs extra is unclear; the submit's `consumed` is the source of truth.

`run` reports `outputs[0].extra.credits` (the submit response's `consumed`). Price it as credits x rate.

| item | credits |
|------|--------:|
| any Gen-2.5 generation | 0.5 |
| tier Gen-2.5-Extreme-High | +0.5 |
| texture_mode extreme-high | +2.0 |
