# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: https://www.context.dev/pricing and https://docs.context.dev/account/credits
- checked: 2026-10-25
- confidence: high: every response reports the exact credits charged in key_metadata.credits_consumed, and the app passes it through

Credit price by plan: Developer $20/month for 10,000 credits ($0.002 each, overage $3.00 per 1,000); Pro $90/month for 100,000 ($0.0009 each, overage $2.00 per 1,000). Failed validation, auth, 408, 429 and server errors are not charged.

The app reports `output_meta.outputs[0]` as a raw item with `extra.credits` = credits the call used, so a price is credits times a per-credit rate.

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| brand | 10 credits | credit | |
| brand_search | 1 credit (free on Pro and up) | credit | |
| styleguide | 10 credits | credit | |
| fonts | see credits_consumed | credit | |
| naics | see credits_consumed | credit | |
| sic | see credits_consumed | credit | |
