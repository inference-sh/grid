# Anthropic Claude — Direct API Provider

Claude models via the native Anthropic API. Zero Data Retention (ZDR) eligible.

## Apps

App names use the dashed model version (`claude-opus-4-6`, not `claude-opus-46`) — the
directory name, the `name:` field in `inf.yml`, and the app in the DB must all match.

| App | Model ID | Context | Max Output | Pricing (in/out per MTok) |
|-----|----------|---------|------------|---------------------------|
| `anthropic/claude-fable-5` | `claude-fable-5` | 1M | 128k | $10 / $50 |
| `anthropic/claude-mythos-5` | `claude-mythos-5` | 1M | 128k | $10 / $50 |
| `anthropic/claude-opus-5` | `claude-opus-5` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-8` | `claude-opus-4-8` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-7` | `claude-opus-4-7` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-6` | `claude-opus-4-6` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-5` | `claude-opus-4-5-20251101` | 200k | 64k | $5 / $25 |
| `anthropic/claude-sonnet-5` | `claude-sonnet-5` | 1M | 64k | $2 / $10 until 2026-08-31, then $3 / $15 |
| `anthropic/claude-sonnet-4-6` | `claude-sonnet-4-6` | 1M | 64k | $3 / $15 |
| `anthropic/claude-sonnet-4-5` | `claude-sonnet-4-5-20250929` | 200k | 64k | $3 / $15 |
| `anthropic/claude-haiku-4-5` | `claude-haiku-4-5-20251001` | 200k | 64k | $1 / $5 |

"Max Output" is the streaming ceiling. **Non-streaming (`stream: false`) is capped at
16k output tokens** (`NON_STREAM_MAX_TOKENS` in `anthropic_helper.py`): the SDK refuses a
non-streaming request whose estimated duration exceeds the client timeout, so a larger
`max_tokens` fails outright with *"Streaming is required for operations that may take
longer than 10 minutes."* Use streaming for long outputs.

## Capabilities

All apps support:
- **Streaming** — Native SSE streaming with token-by-token output
- **Vision** — Image input via URL or base64 (png, jpg, gif, webp)
- **Reasoning** — See the thinking-parameter table below; the shape is model-dependent
- **Tool use** — Function calling in Anthropic native format
- **Token tracking** — Input/output token counts via `output_meta`

### Thinking parameter by model

`anthropic_helper.py` picks the shape per model — these are not interchangeable and the
wrong one is a hard 400, not a degradation:

| Models | Shape |
|--------|-------|
| `claude-fable-5`, `claude-mythos-5` | Parameter **omitted** — thinking is always on; an explicit `{"type": "disabled"}` returns 400 |
| `claude-opus-5` | Always `{"type": "adaptive"}` + `output_config.effort`. Never disabled: with thinking off, the model can emit a tool call as plain response text (the call silently never runs) and can leak `<thinking>` tags. Cost is controlled with a lower effort instead. |
| `claude-opus-4-8`, `claude-opus-4-7`, `claude-sonnet-5` | `{"type": "adaptive"}` + `output_config.effort`, or `{"type": "disabled"}`. **`budget_tokens` returns 400.** |
| `claude-opus-4-6` and earlier, all Sonnet 4.x, Haiku 4.5 | `{"type": "enabled", "budget_tokens": N}` or `{"type": "disabled"}` |

Adding a model to `EFFORT_MODELS` in the helper switches it from `budget_tokens` to
adaptive + effort. **Redeploy every app after editing the helper** — it is symlinked, so a
deployed app keeps the copy bundled at its own deploy time and does not pick up edits.

## Auth

Uses `ANTHROPIC_KEY` secret (set via `belt secrets set ANTHROPIC_KEY <your-key>`).

## Pricing

Rates and the CEL setup live in `pricing.md`. One trap worth repeating here: price
variables are denominated in **microcents**, where $1 = 100,000,000. So $5/MTok is
`500000000`, not `5000000`. The pricing agent has produced the 100x-too-low value more
than once — always check `belt app pricing <ns/app>` against a sibling app before
approving a publish.

## Not yet available (models don't exist)

- `claude-sonnet-4-7` — no Sonnet 4.7 release (the line went 4.6 → 5)
- `claude-haiku-4-6`, `claude-haiku-4-7`, `claude-haiku-5` — no Haiku release past 4.5

## Bedrock / Vertex / AWS (planned)

Separate apps will be created for cloud provider variants:
- `anthropic/claude-opus-47-bedrock` — AWS Bedrock (`anthropic.claude-opus-4-7`)
- `anthropic/claude-opus-47-vertex` — Google Vertex AI (`claude-opus-4-7`)

These require different auth (AWS IAM / Google Cloud credentials) so they will be separate apps with a setup option for importing credentials conditionally.

### Bedrock Model IDs
| Model | Bedrock ID |
|-------|-----------|
| Opus 4.7 | `anthropic.claude-opus-4-7` |
| Sonnet 4.6 | `anthropic.claude-sonnet-4-6` |
| Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| Opus 4.6 | `anthropic.claude-opus-4-6-v1` |
| Sonnet 4.5 | `anthropic.claude-sonnet-4-5-20250929-v1:0` |

### Vertex AI Model IDs
| Model | Vertex ID |
|-------|----------|
| Opus 4.7 | `claude-opus-4-7` |
| Sonnet 4.6 | `claude-sonnet-4-6` |
| Haiku 4.5 | `claude-haiku-4-5@20251001` |
| Opus 4.6 | `claude-opus-4-6` |
| Sonnet 4.5 | `claude-sonnet-4-5@20250929` |

## API Reference

- Base URL: `https://api.anthropic.com`
- Endpoint: `POST /v1/messages`
- Auth header: `x-api-key`
- Version header: `anthropic-version: 2023-06-01`
- Request size limit: 32 MB
- Docs: https://platform.claude.com/docs/en/api/overview
