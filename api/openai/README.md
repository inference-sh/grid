# OpenAI — Direct API Provider

OpenAI models via the native OpenAI API. Two families live here:

- **Chat / LLM apps** (`gpt-6-astra`, `gpt-5-6-*`) — share `openai_llm.py`, stream the
  Responses API over raw SSE and yield `LLMDelta` chunks (INF-630 delta streaming).
- **Image apps** (`gpt-image-2`) — share `openai_helper.py` and use the OpenAI SDK.

Both helpers are symlinked into each app directory. Editing a helper changes every app that
links it — **redeploy all of them**, a deployed app keeps the copy bundled at deploy time.

## LLM apps

App names use dashes only (`gpt-5-6-sol`, not `gpt-5.6-sol`). Directory name, `name:` in
`inf.yml`, and the app in the DB must match.

| App | Model ID | Context | Max Output | Reasoning effort | Pricing (in / cached / out per MTok) |
|-----|----------|---------|------------|------------------|--------------------------------------|
| `openai/gpt-6-astra` | `gpt-6-astra` | 1.05M | 128k | low, medium, high, xhigh, max — **no `none`** | $10 / $1 / $50 |
| `openai/gpt-5-6-sol` | `gpt-5.6-sol` | 1.05M | 128k | none, low, medium, high, xhigh, max | $4 / $0.40 / $20 |
| `openai/gpt-5-6-terra` | `gpt-5.6-terra` | 1.05M | 128k | none, low, medium, high, xhigh, max | $2 / $0.20 / $12 |
| `openai/gpt-5-6-luna` | `gpt-5.6-luna` | 1.05M | 128k | none, low, medium, high, xhigh, max | $0.20 / $0.02 / $1.20 |

Source: https://developers.openai.com/api/docs/pricing and the per-model pages under
https://developers.openai.com/api/docs/models/. Fast mode is 2x these rates and is not
exposed. Prompts above 272k input tokens are billed at 2x input / 1.5x output by OpenAI;
that tier is not modelled in our pricing.

### How the LLM apps work

- **Delta streaming.** `run` is an async generator that yields `LLMDelta` per SSE event
  (`response` text, `reasoning` summary text, `tool_calls` argument fragments with `index`)
  and one final `AppOutput` carrying the full response, `usage` and `output_meta`.
- **Responses API, not Chat Completions.** Chosen because it is the only endpoint that
  exposes reasoning summaries (`reasoning.summary: "auto"`). If the org is not verified
  for summaries the API returns 400; the helper drops `summary` and retries, then keeps it
  off for the life of the worker.
- **Reasoning effort mapping.** The platform `reasoning_effort` enum is `none | low |
  medium | high` (default `none`). GPT-6 Astra rejects `none` (HTTP 400), so
  `SUPPORTS_NONE_REASONING = False` in that app maps `none` to `low`. `xhigh` and `max`
  are not reachable through the platform enum.
- **No sampling params.** `temperature` / `top_p` are rejected by reasoning models and
  are never sent, whatever the input carries.
- **Tool calls round-trip on `call_id`.** Assistant `tool_calls[].id` is the Responses
  `call_id`. When context is replayed, `function_call` items are sent **without** the
  `fc_…` item id — with it present the API demands the paired reasoning item, which we
  do not persist.
- **Files and images** are passed by URL when they are `http(s)` URLs, else base64.
- **Timeouts.** Connect/write only. There is no read timeout on the stream: at `max`
  effort the API can be silent for minutes before the first event. The platform owns
  task-level timeouts.

### Token metering (`output_meta`)

```
inputs[0]  = TextMeta(tokens=<input_tokens, cached included>, extra={"cache_read_tokens": N})
outputs[0] = TextMeta(tokens=<output_tokens, reasoning included>, extra={"reasoning_tokens": N})
```

OpenAI's `input_tokens` already includes cached tokens, so the uncached count is
`text_tokens(inputs) - cache_read_tokens`. Reasoning tokens are billed as ordinary output
tokens and are already inside `output_tokens`.

### Pricing (for the pricing agent)

All four are third-party LLM wrappers billed on tokens. Price variables are in
**microcents** ($1 = 100,000,000). Expected values:

```
# gpt-6-astra: $10 in, $1 cached, $50 out
partner_input_per_million:       1000000000
partner_cache_read_per_million:   100000000
partner_output_per_million:      5000000000

# gpt-5-6-sol: $4 in, $0.40 cached, $20 out
partner_input_per_million:        400000000
partner_cache_read_per_million:    40000000
partner_output_per_million:      2000000000

# gpt-5-6-terra: $2 in, $0.20 cached, $12 out
partner_input_per_million:        200000000
partner_cache_read_per_million:    20000000
partner_output_per_million:      1200000000

# gpt-5-6-luna: $0.20 in, $0.02 cached, $1.20 out
partner_input_per_million:         20000000
partner_cache_read_per_million:     2000000
partner_output_per_million:       120000000
```

Anything with two fewer zeros is the 100x unit error. Verify with
`belt app pricing openai/<app>` against a sibling before approving a publish.

## Auth

All apps use the `OPENAI_KEY` secret (`belt secrets set OPENAI_KEY <key>`).

## Dev loop

```bash
cd api/openai/gpt-6-astra
belt app deploy --dry-run                 # validate + regenerate schemas
belt app deploy
belt app run openai/gpt-6-astra --json --input '{"text": "hi"}'
```

Offline check of the helper (no API key needed): build a request body with
`build_request_body(...)` and fold synthetic events through `_handle_event(...)`, then
validate the deltas with `LLMDelta(**delta)`.
