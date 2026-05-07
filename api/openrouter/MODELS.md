# OpenRouter Models

Top models by usage on OpenRouter (May 2026 leaderboard). Pricing is per million tokens.

## Leaderboard Top 10 (This Month)

| # | App Dir | Model ID | Modality | Context | Input $/M | Output $/M | Capabilities | Status |
|---|---------|----------|----------|---------|-----------|------------|--------------|--------|
| 1 | `claude-sonnet-46` | `anthropic/claude-sonnet-4.6` | text+image→text | 1M | $3.00 | $15.00 | reasoning, image | **NEW** |
| 2 | `hy3-preview` | `tencent/hy3-preview:free` | text→text | 262k | free | free | reasoning | **NEW** |
| 3 | `deepseek-v32` | `deepseek/deepseek-v3.2` | text→text | 131k | $0.252 | $0.378 | reasoning, image, file | exists |
| 4 | `gemini-3-flash-preview` | `google/gemini-3-flash-preview` | text+image+file+audio+video→text | 1M | $0.50 | $3.00 | reasoning, image, file | **NEW** |
| 5 | `kimi-k26` | `moonshotai/kimi-k2.6` | text+image→text | 262k | $0.75 | $3.50 | reasoning, image | **NEW** |
| 6 | `minimax-m-27` | `minimax/minimax-m2.7` | text→text | 196k | $0.30 | $1.20 | reasoning | exists |
| 7 | `claude-opus-46` | `anthropic/claude-opus-4.6` | text+image→text | 1M | $5.00 | $25.00 | reasoning, image, file | exists |
| 8 | `minimax-m-25` | `minimax/minimax-m2.5` | text→text | 196k | $0.15 | $1.15 | reasoning | exists |
| 9 | `claude-opus-47` | `anthropic/claude-opus-4.7` | text+image→text | 1M | $5.00 | $25.00 | reasoning, image | **NEW** |
| 10 | `grok-41-fast` | `x-ai/grok-4.1-fast` | text+image+file→text | 2M | $0.20 | $0.50 | reasoning, image, file | **NEW** |

## Most Expensive (per M output tokens)

1. `anthropic/claude-opus-4.6` — $25.00
2. `anthropic/claude-opus-4.7` — $25.00
3. `anthropic/claude-sonnet-4.6` — $15.00
4. `moonshotai/kimi-k2.6` — $3.50
5. `google/gemini-3-flash-preview` — $3.00
6. `minimax/minimax-m2.7` — $1.20
7. `minimax/minimax-m2.5` — $1.15
8. `x-ai/grok-4.1-fast` — $0.50
9. `deepseek/deepseek-v3.2` — $0.378
10. `tencent/hy3-preview:free` — free

## Capability → Mixin Mapping

| Modality | Mixins |
|----------|--------|
| `text→text` | `ReasoningCapabilityMixin`, `ToolsCapabilityMixin` |
| `text+image→text` | + `ImageCapabilityMixin` |
| `text+image+file→text` | + `ImageCapabilityMixin`, `FileCapabilityMixin` |
| `text+image+file+audio+video→text` | + `ImageCapabilityMixin`, `FileCapabilityMixin` |
