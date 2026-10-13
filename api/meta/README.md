# Meta — Direct API Provider

Meta Model API (dev.meta.ai, base URL `https://api.meta.ai/v1`). The chat apps share
`meta_llm.py`, stream the Responses API over raw SSE and yield `LLMDelta` chunks.

The helper is symlinked into each app directory. Editing it changes every app that links
it — **redeploy all of them**, a deployed app keeps the copy bundled at deploy time.

## LLM apps

App names use dashes only (`muse-wonder-1-0` for `muse-wonder-1.0`). Directory name,
`name:` in `inf.yml`, and the app in the DB must match.

| App | Model ID | Context | Reasoning effort | Pricing (in / cached / out per MTok) |
|-----|----------|---------|------------------|--------------------------------------|
| `meta/muse-wonder-1-0` | `muse-wonder-1.0` | 1,048,576 | minimal, low, medium, high, xhigh, max — **no `none`** | $4 / $0.20 / $20 |

Source: https://dev.meta.ai/docs/pricing and https://dev.meta.ai/docs/models. Muse Wonder
is in preview and has no Contributor mode on Model API. Rate limit: 3,000 RPM, 4M TPM per team.

### How the LLM apps work

- **Delta streaming.** `run` is an async generator that yields `LLMDelta` per SSE event
  (`response` text, `reasoning` summary text, `tool_calls` argument fragments with `index`)
  and one final `AppOutput` carrying the full response, `usage` and `output_meta`.
- **Responses API, not Chat Completions.** It is the only endpoint with reasoning summaries
  (`reasoning.summary: "auto"`) and encrypted reasoning for stateless replay. Raw chain of
  thought is never returned as text.
- **Reasoning effort mapping.** `none` is HTTP 400 on every Muse model, so the platform
  default maps to the nearest accepted value (`minimal`). Each app lists its accepted set
  in `REASONING_EFFORTS`; a value outside it maps to the nearest one.
- **Reasoning across turns.** Requests run with `store: false` and
  `include: ["reasoning.encrypted_content"]`; the reasoning items come back on
  `reasoning_state.meta.items` and are replayed before the turn's text and calls.
  A replayed item always carries `summary` (`[]` when none), which the API requires.
- **max_output_tokens** is the caller's `max_tokens` (platform default 64000; minimum 16,
  capped at `MAX_OUTPUT_TOKENS`). Reasoning shares that budget.
- **tool_choice.** Only `auto` works upstream. `none` is honoured by not sending tools;
  `required` and a named function are refused before the request with a `ValueError`.
- **Tool results.** `function_call_output` is a string. Meta reads images only in user
  messages, so a tool result with images or files is followed by a user message
  "Attachments returned by tool call <id>" carrying them.
- **Media goes inline.** Meta fetches `image_url` and `file_url` itself and refuses an
  origin whose robots.txt disallows it (`400 media_url_robots_disallowed`). The platform
  has already downloaded every input, so the local copy is sent as a data URL; a URL is
  passed on only when there is no local copy.
- **Images** go as `input_image` (no `detail` parameter). JPEG, PNG, GIF, WebP, ICO; 50 MB
  each; 50 per request (refused client-side above that).
- **Files** go as `input_file` with `file_data`. Documented types: pdf, mp4, mp3, wav, txt,
  json. PDFs: text from the first 100 pages, page images from the first 50.
- **Sampling.** `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` are sent
  only when the caller set them (the LLMInput defaults 0.7 / 0.95 are not). `top_k`, `seed`, `stop` are not supported and never sent.
- **Capacity shows up as a 404.** Meta sheds load on its newest model with
  `404 model_not_found` instead of a 503: same key, worker, connection and request flip
  between 200 and 404 within seconds, the 404 comes from the router in ~400 ms, and
  `/v1/models` keeps listing the model. Seen on muse-spark-1.3 in September
  ([CLIProxyAPI #6117](https://github.com/router-for-me/CLIProxyAPI/issues/6117),
  [popcre/ai-devops #743](https://github.com/popcre/ai-devops/issues/743)) and on
  muse-wonder-1.0 the day after launch. The helper turns that exact body into a
  `503 service_overloaded` before anything reads it, so it is retried and reported as one.
- **Retries.** Connection errors, 408/429/500/502/503 (honouring `Retry-After`) and a
  stream-level `error` with code `server_shutting_down` before any output. No read timeout
  on the stream: the platform owns task-level timeouts.

### Token metering (`output_meta`)

```
inputs[0]  = TextMeta(tokens=<input_tokens, cached included>, cached_tokens=N)
outputs[0] = TextMeta(tokens=<output_tokens, reasoning included>)
```

Meta's `input_tokens` already includes cached tokens; reasoning tokens are a detail of
`output_tokens` and billed at the output rate. If a response ever reports them separately,
`total_tokens` settles it (`billable_output_tokens`).

### Pricing (for the pricing agent)

Price variables are in **microcents** ($1 = 100,000,000). See `pricing.md`.

## Auth

All apps use the `META_AI_KEY` secret (`belt secrets set META_AI_KEY <key>`), an API key from the
dev.meta.ai dashboard.

## Dev loop

```bash
cd api/meta
PYTHONPATH=../../../sdk-py/src python3 -m pytest test_meta_llm.py -q   # no network

cd muse-wonder-1-0
belt app deploy --dry-run                 # validate + regenerate schemas
belt app deploy
belt app run meta/muse-wonder-1-0 --json --input '{"text": "hi"}'
```
