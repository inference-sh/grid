# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: https://www.context.dev/pricing and https://docs.context.dev/account/credits
- checked: 2026-10-25
- confidence: high: the credit table is published per operation; each call logs the credits context.dev reports (key_metadata.credits_consumed) so the price can be checked against it

Credit price by plan: Developer $20/month for 10,000 credits ($0.002 each, overage $3.00 per 1,000); Pro $90/month for 100,000 ($0.0009 each, overage $2.00 per 1,000). Validation, auth, 408, 429 and server errors are not charged.

Price from the input and from what the call returned, never from the vendor's reported charge. A function whose cost depends on its result reports the count as a raw output item, `output_meta.outputs[0].extra`. The same item carries `upstream_credits`, the credits context.dev reports for the call: informational only, never used in a price.

| function | upstream rate | priced from | notes |
|----------|---------------|-------------|-------|
| brand | 10 credits | flat | |
| brand_search | 1 credit (free on Pro and up) | flat | |
| styleguide | 10 credits | flat | |
| fonts | 5 credits | flat | |
