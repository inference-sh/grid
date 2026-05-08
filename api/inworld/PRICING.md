# Inworld AI API Pricing

## Text to Speech

| Model | Price per 1M chars |
|-------|-------------------|
| TTS-2 | $35 |
| TTS 1.5 Max | $35 |
| TTS 1.5 Mini | $25 |

Models:
- `inworld-tts-2` - Highest quality, 100+ languages, delivery mode steering
- `inworld-tts-1.5-max` - Low latency (<200ms P50), 15 languages
- `inworld-tts-1.5-mini` - Ultra-low latency (~120ms P50), 15 languages
- 2,000 character limit per request

## Speech to Text

| Model | Price per hour |
|-------|---------------|
| Inworld STT | $0.35 |

Models:
- `inworld/inworld-stt-1` - First-party STT, sync + websocket
- `groq/whisper-large-v3` - Groq Whisper, sync only

## LLMs

At cost (pass-through pricing).

---

## Price Variables (microcents)

```
# TTS - per 1M characters (= per 1K chars pricing below)
# TTS-2 & 1.5 Max: $35/1M chars = $0.035/1K chars
TTS_2 = 3_500_000            # $0.035 per 1K chars
TTS_1_5_MAX = 3_500_000      # $0.035 per 1K chars

# TTS 1.5 Mini: $25/1M chars = $0.025/1K chars
TTS_1_5_MINI = 2_500_000     # $0.025 per 1K chars

# STT - per hour
STT = 35_000_000              # $0.35
```

## CEL Expression Examples

### TTS (per 1K characters, model-based pricing)
```cel
// outputs[0].extra.characters contains char count
// outputs[0].extra.model contains model name
(double(outputs[0].extra.characters) / 1000.0) * double(prices.per_1k_characters)
```

### STT (per hour of audio)
```cel
// inputs[0].seconds contains duration
// inputs[0].extra.model contains model name
(double(inputs[0].seconds) / 3600.0) * double(prices.per_hour)
```
