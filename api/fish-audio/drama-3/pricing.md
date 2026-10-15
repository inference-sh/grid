# Pricing

What Fish Audio charges for `drama-3`, so the app's price can be set from it.

- source_url: https://fish.audio/blog/fish-audio-drama-3/ (drama-3 is not yet on https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits)
- checked: 2026-10-15
- confidence: medium. The launch post gives the rate per million characters; the pricing page bills the other models per million UTF-8 bytes. Confirm the unit on the first invoice.
- published: list price, `prices.per_1m_bytes = 5000000000`, on version 678h4stx. Task 0a39p8ffwnxcz8f1ktm42twqw0 (107 bytes) billed 535,000 microcents.

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| run | $50.00 (list) | 1M characters (UTF-8 bytes?) | $10.00 during the launch offer, 2026-10-13 to 2026-11-12 |

For comparison, s2.1-pro is $15.00 per 1M UTF-8 bytes.

`run` reports `outputs[0].extra.utf8_bytes` and `outputs[0].extra.characters` (the input text), plus the audio's `seconds`.

Expected microcents at list price ($50/M = 5,000 microcents per byte):

| input | utf8_bytes | microcents |
|-------|-----------:|-----------:|
| test_requests/run.json | 107 | 535,000 |
| 1,000 ASCII characters | 1,000 | 5,000,000 |
