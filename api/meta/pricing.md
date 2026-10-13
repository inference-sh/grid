# Meta Model API Pricing

Pay-as-you-go, Standard mode. Source: https://dev.meta.ai/docs/pricing (2026-10-13).
All prices in USD per 1M tokens. No long-context premium. Web search grounding
($2.50 per 1,000 queries) is not exposed by these apps.

## Muse Wonder

| Model | Input | Cached input | Output |
|-------|-------|--------------|--------|
| `muse-wonder-1.0` | $4.00 | $0.20 | $20.00 |

Contributor mode is not available for Muse Wonder on Model API.

## Muse Spark (not yet wrapped)

| Models | Mode | Input | Cached input | Output |
|--------|------|-------|--------------|--------|
| `muse-spark-1.3`, `muse-spark-1.2`, `muse-spark-1.1` | Standard | $1.25 | $0.15 | $4.25 |
| `muse-spark-1.3-contributor`, `muse-spark-1.2-contributor` | Contributor | $0.10 | $0.002 | $0.20 |

## OutputMeta mapping

- `inputs[0].tokens` — input tokens, cached included
- `inputs[0].cached_tokens` — tokens served from the prefix cache (`usage.input_tokens_details.cached_tokens`)
- `outputs[0].tokens` — output tokens, reasoning included

Uncached input = `inputs[0].tokens - inputs[0].cached_tokens`.

## Microcents ($1 = 100,000,000)

```
# muse-wonder-1-0: $4 in, $0.20 cached, $20 out
partner_input_per_million:        400000000
partner_cache_read_per_million:    20000000
partner_output_per_million:      2000000000
```

Anything with two fewer zeros is the 100x unit error. Verify with
`belt app pricing meta/<app>` against `openai/gpt-5-6-sol` ($4 / $0.40 / $20) before approving.
