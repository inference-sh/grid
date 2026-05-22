# OpenRouter Provider

LLM apps via [OpenRouter](https://openrouter.ai) — one shared helper, one app per model.

## Adding a new model

```bash
./or-scaffold.sh <model-id> [app-dir]
```

Example:

```bash
./or-scaffold.sh anthropic/claude-sonnet-4.6 claude-sonnet-46
```

This will:

1. Run `belt app init <app-dir>` to create the proper app skeleton
2. Fetch the model's capabilities, pricing, and context length from the OpenRouter API
3. Overlay `inference.py`, `inf.yml`, `__init__.py`, and `requirements.txt` with the right mixins for the model's modality
4. Symlink the shared `openrouter.py` helper
5. Generate `MODEL.md` with pricing and supported parameters

If `app-dir` is omitted, it derives one from the model ID (e.g. `anthropic/claude-sonnet-4.6` becomes `claude-sonnet-46`).

## How it works

All apps share a single `openrouter.py` helper (symlinked into each app dir) that handles streaming, tool calls, reasoning, and usage tracking. Each app is just ~60 lines that set `DEFAULT_MODEL` and declare the right input mixins.

The scaffold auto-detects modality from the API and picks mixins accordingly:

| Modality | Extra Mixins |
|----------|-------------|
| `text->text` | _(none)_ |
| `text+image->text` | `ImageCapabilityMixin` |
| `text+image+file->text` | `ImageCapabilityMixin`, `FileCapabilityMixin` |
| `text+image+file+audio+video->text` | `ImageCapabilityMixin`, `FileCapabilityMixin` |

## After scaffolding

1. Review `inf.yml` — update description and add card/thumbnail/banner images
2. Deploy: `cd <app-dir> && belt app deploy`
3. Test: `belt app run openrouter/<app-dir> --json --input '{"prompt":"hello"}'`

## Structure

```
openrouter/
├── openrouter.py          # Shared helper (streaming, tools, reasoning, usage)
├── or-scaffold.sh         # Scaffold script
├── MODELS.md              # Leaderboard pricing reference
├── README.md
├── claude-sonnet-46/      # One dir per model
│   ├── inference.py
│   ├── inf.yml
│   ├── __init__.py
│   ├── requirements.txt
│   ├── openrouter.py      # -> ../openrouter.py
│   └── MODEL.md
└── ...
```

## Sampling parameters

**Always send explicit sampling params.** OpenRouter is a passthrough router — if you omit
`temperature`, `top_p`, `top_k`, or `min_p`, the routed provider fills in its own default.
Different providers (e.g. AtlasCloud vs Alibaba for Qwen) default differently, which silently
breaks reproducibility. The shared `openrouter.py` helper now forwards all sampling params from
the app's `AppInput` to the API.

### Where to get the right values

1. **Model vendor's card** — authoritative. The vendor tuned the model with these values.
2. **Paper / upstream repo** — for reproductions.
3. **Generic fallbacks** (temp 0.7, top_p 0.9) — only when nothing model-specific exists.

### Per-model recommendations

Override defaults in each app's `AppInput` class. The shared helper forwards them.

| Model | temperature | top_p | top_k | min_p | Source |
|-------|------------|-------|-------|-------|--------|
| **Qwen3 (thinking)** | 0.6 | 0.95 | 20 | 0 | [HF model card](https://huggingface.co/Qwen/Qwen3-8B) |
| **Qwen3 (non-thinking)** | 0.7 | 0.8 | 20 | 0 | [HF model card](https://huggingface.co/Qwen/Qwen3-8B) |
| **DeepSeek V3** | 0.0–0.7 | 0.95 | — | — | [DeepSeek docs](https://api-docs.deepseek.com) |
| **Claude** | 1.0 | — | — | — | Anthropic default; top_k/min_p not applicable |
| **Gemini** | 1.0 | 0.95 | 40 | — | Google AI Studio defaults |
| **Kimi K2** | 0.6 | 0.95 | — | — | Moonshot docs |

Qwen3 apps auto-switch between thinking/non-thinking defaults via the `_qwen_nothink_hook`
in `openrouter.py`. For other models, add hooks in `_MODEL_HOOKS` as needed.

### Adding sampling fields to a new app

```python
class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    # Override LLMInput defaults with vendor-recommended values
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1, description="Top-k sampling. -1 to disable.")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p sampling threshold.")
```

The shared helper reads these via `getattr` — if a field isn't defined, it falls back to
`BaseLLMInput` defaults (temp=0.7, top_p=0.95) or skips the param (top_k, min_p).

## See also

- [MODELS.md](MODELS.md) — current leaderboard top 10 with pricing
