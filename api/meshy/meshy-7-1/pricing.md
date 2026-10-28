# Pricing

What Meshy charges for `meshy-7.1`, so the app's price can be set from it.

- source_url: https://docs.meshy.ai/api/pricing (credits); https://www.meshy.ai/pricing (plans)
- checked: 2026-10-28
- confidence: medium. Credits per task are published; dollars per credit depend on the plan (Pro $0.020, Premium ~$0.0133, Ultra $0.0125). Set the rate from the plan the platform key is on.

`run` reports `outputs[0].extra.credits`: the sum of `consumed_credits` Meshy returned for the run's tasks (a textured text run is a preview plus a refine). Price it as credits x rate.

| mode | mesh only | 2K/4K texture | 8K texture |
|------|----------:|--------------:|-----------:|
| text (preview + refine) | 20 | 30 | 35 |
| image | 20 | 30 | 35 |
| multi-image | 20 | 30 | 35 |

`geometry_resolution` 2k or 4k adds 5 credits.
