# Anthropic Claude — Direct API Provider

Claude models via the native Anthropic API. Zero Data Retention (ZDR) eligible.

## Apps

App names use the dashed model version (`claude-opus-4-6`, not `claude-opus-46`) — the
directory name, the `name:` field in `inf.yml`, and the app in the DB must all match.

| App | Model ID | Context | Max Output | Pricing (in/out per MTok) |
|-----|----------|---------|------------|---------------------------|
| `anthropic/claude-fable-5-1` | `claude-fable-5-1` | 1M | 128k | $10 / $50 (cache read $0.25) |
| `anthropic/claude-fable-5` | `claude-fable-5` | 1M | 128k | $10 / $50 |
| `anthropic/claude-mythos-5` | `claude-mythos-5` | 1M | 128k | $10 / $50 |
| `anthropic/claude-opus-5-5` | `claude-opus-5-5` | 1M | 128k | $4 / $20 (cache read $0.20) |
| `anthropic/claude-opus-5` | `claude-opus-5` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-8` | `claude-opus-4-8` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-7` | `claude-opus-4-7` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-6` | `claude-opus-4-6` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-4-5` | `claude-opus-4-5-20251101` | 200k | 64k | $5 / $25 |
| `anthropic/claude-sonnet-5-5` | `claude-sonnet-5-5` | 1M | 128k | $2 / $10 (cache read $0.20) |
| `anthropic/claude-sonnet-5` | `claude-sonnet-5` | 1M | 64k | $2 / $10 until 2026-08-31, then $3 / $15 |
| `anthropic/claude-sonnet-4-6` | `claude-sonnet-4-6` | 1M | 64k | $3 / $15 |
| `anthropic/claude-sonnet-4-5` | `claude-sonnet-4-5-20250929` | 200k | 64k | $3 / $15 |
| `anthropic/claude-haiku-5-5` | `claude-haiku-5-5` | 1M | 128k | $0.10 / $0.50 up to 100k input tokens, $0.50 / $2.50 above |
| `anthropic/claude-haiku-4-5` | `claude-haiku-4-5-20251001` | 200k | 64k | $1 / $5 |

"Max Output" is the streaming ceiling and the cap on the caller's `max_tokens` (LLMInput
defaults it to 64000); thinking shares that budget. **Non-streaming (`stream: false`) is capped at
16k output tokens** (`NON_STREAM_MAX_TOKENS` in `anthropic_helper.py`): the SDK refuses a
non-streaming request whose estimated duration exceeds the client timeout, so a larger
`max_tokens` fails outright with *"Streaming is required for operations that may take
longer than 10 minutes."* Use streaming for long outputs.

## Capabilities

All apps support:
- **Streaming** — Native SSE streaming with token-by-token output
- **Vision** — Image input via URL or base64 (png, jpg, gif, webp)
- **Files** — `files` input (`file` capability): a PDF goes as a `document` block (base64 from the
  downloaded file, or by URL; the API takes up to 32MB and 100 pages), a text file (md, csv, json,
  source...) as a plain-text document. Any other type is named to the model in a text block, since
  the API cannot read it; the model then knows a file was sent instead of denying it
- **Reasoning** — See the thinking-parameter table below; the shape is model-dependent
- **Tool use** — Function calling in Anthropic native format
- **Token tracking** — Input/output token counts via `output_meta`

### Thinking parameter by model

`anthropic_helper.py` picks the shape per model — these are not interchangeable and the
wrong one is a hard 400, not a degradation:

| Models | Shape |
|--------|-------|
| `claude-fable-5`, `claude-mythos-5` | `{"type": "adaptive", "display": "summarized"}` — thinking is always on; an explicit `{"type": "disabled"}` returns 400 |
| `claude-fable-5-1`, `claude-mythos-5-1`, `claude-opus-5-5` | `{"type": "adaptive", "display": "summarized"}` + `output_config.effort` (`low` when no reasoning is asked for; Opus 5.5's own default is `medium`). `disabled` and `budget_tokens` are 400 at every effort. Forced `tool_choice` (`required` / a named function) is a 400 and is refused before the request. |
| `claude-sonnet-5-5` | `{"type": "adaptive"}` + `output_config.effort`, or `{"type": "between_tools"}` when no reasoning is asked for (`disabled` is a 400; `between_tools` takes no other field). Forced `tool_choice` refused before the request. |
| `claude-haiku-5-5` | `{"type": "adaptive"}` + `output_config.effort`, or `{"type": "disabled"}` (allowed at effort `high` or below, which every platform level is). `budget_tokens` returns 400. Forced `tool_choice` still accepted. |
| `claude-opus-5` | Always `{"type": "adaptive", "display": "summarized"}` + `output_config.effort`. Never disabled: with thinking off, the model can emit a tool call as plain response text (the call silently never runs) and can leak `<thinking>` tags. Cost is controlled with a lower effort instead. |
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

## Lifecycle

- `claude-sonnet-4-5-20250929` is deprecated (2026-09-30), retires 2026-11-30, replacement `claude-sonnet-5-5`.
- `claude-haiku-4-5-20251001` retires not sooner than 2026-10-15, `claude-opus-4-5-20251101` not sooner than 2026-11-24.
- `claude-mythos-5-1` exists (same price as Fable 5.1, Project Glasswing only); the workspace key is not entitled to the Mythos line, so there is no app.
- `usage.stop_reason` is Anthropic's stop reason in platform terms: `end_turn` / `stop_sequence` → `stop`, `tool_use` → `tool_calls`, `max_tokens` → `length`; `refusal` (safety classifiers on the 5.1 / 5.5 models) and `pause_turn` pass through.
- Adaptive thinking is requested with `display: "summarized"`: Opus 4.7 and later default to omitted (empty thinking text), and the summary is what `reasoning` carries. Billing is the same either way; Anthropic reports no separate reasoning token count, so `reasoning_tokens` stays 0 and thinking is inside `completion_tokens`.

## Not yet available (models don't exist)

- `claude-sonnet-4-7` — no Sonnet 4.7 release (the line went 4.6 → 5)
- `claude-haiku-4-6`, `claude-haiku-4-7`, `claude-haiku-5` — the Haiku line went 4.5 → 5.5

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
