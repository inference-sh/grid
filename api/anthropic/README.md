# Anthropic Claude — Direct API Provider

Claude models via the native Anthropic API. Zero Data Retention (ZDR) eligible.

## Apps

| App | Model ID | Context | Max Output | Pricing (in/out per MTok) |
|-----|----------|---------|------------|---------------------------|
| `anthropic/claude-opus-47` | `claude-opus-4-7` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-46` | `claude-opus-4-6` | 1M | 128k | $5 / $25 |
| `anthropic/claude-opus-45` | `claude-opus-4-5-20251101` | 200k | 64k | $5 / $25 |
| `anthropic/claude-sonnet-46` | `claude-sonnet-4-6` | 1M | 64k | $3 / $15 |
| `anthropic/claude-sonnet-45` | `claude-sonnet-4-5-20250929` | 200k | 64k | $3 / $15 |
| `anthropic/claude-haiku-45` | `claude-haiku-4-5-20251001` | 200k | 64k | $1 / $5 |

## Capabilities

All apps support:
- **Streaming** — Native SSE streaming with token-by-token output
- **Vision** — Image input via URL or base64 (png, jpg, gif, webp)
- **Extended thinking** — Configurable reasoning with budget tokens
- **Tool use** — Function calling in Anthropic native format
- **Token tracking** — Input/output token counts via `output_meta`

## Auth

Uses `ANTHROPIC_KEY` secret (set via `belt secrets set ANTHROPIC_KEY <your-key>`).

## Not yet available (models don't exist)

- `claude-sonnet-47` — No Sonnet 4.7 released yet
- `claude-haiku-46` — No Haiku 4.6 released yet
- `claude-haiku-47` — No Haiku 4.7 released yet

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
