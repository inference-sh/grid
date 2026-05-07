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

## See also

- [MODELS.md](MODELS.md) — current leaderboard top 10 with pricing
