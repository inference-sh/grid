# ElevenLabs API Pricing

Based on Starter plan pricing (we charge at this tier).

## Text to Speech

| Model | Price per 1K chars |
|-------|-------------------|
| Flash / Turbo | $0.15 |
| Multilingual v2 / v3 | $0.30 |

Models:
- `eleven_flash_v2_5` - Ultra-low latency (~75ms), Flash tier
- `eleven_flash_v2` - Flash tier
- `eleven_turbo_v2_5` - Low latency, Turbo tier
- `eleven_turbo_v2` - Turbo tier
- `eleven_multilingual_v2` - Highest quality, ~250-300ms
- `eleven_v3` - Latest multilingual model
- 40,000 character limit per request

## Speech to Text (Scribe)

| Model | Price per hour |
|-------|---------------|
| scribe_v1 | $0.48 |
| scribe_v2 | $0.48 |

- 98%+ transcription accuracy
- 90+ languages
- Optional add-ons:
  - Keyterm prompting: +$0.070/hour
  - Entity detection: +$0.105/hour

## Music Generation

| Service | Price |
|---------|-------|
| Music compose | $0.48 per minute |

- Max 5 minute duration per request
- Commercial licensing included on Starter+
- 44.1kHz, 128-192kbps audio

## Voice Isolator

| Service | Price |
|---------|-------|
| Audio isolation | $0.30 per minute |

- Removes ambient sounds, reverb, interference
- WAV, MP3, FLAC, OGG, AAC inputs
- Max 500MB / 1 hour per file

## Voice Changer (Speech to Speech)

| Model | Price |
|-------|-------|
| eleven_multilingual_sts_v2 | $0.30 per minute |
| eleven_english_sts_v2 | $0.30 per minute |

- Real-time processing
- 10,000+ human-like voices
- 70+ languages

## Sound Effects

| Service | Price |
|---------|-------|
| Sound generation | $0.18 per generation |

- Text-to-sound-effect
- Royalty-free
- MP3 (44.1kHz) or WAV (48kHz) output

## Text to Dialogue

| Model | Price per 1K chars |
|-------|-------------------|
| eleven_v3 | $0.30 |

- Same pricing as Multilingual TTS (per character across all dialogue lines)
- Multi-speaker dialogue in single request
- Total character count = sum of all text fields

## Forced Alignment

| Service | Price per hour |
|---------|---------------|
| Forced alignment | $0.48 |

- Same pricing as STT
- Word-level and character-level timestamps
- Useful for subtitles, lip-sync, karaoke

## Dubbing

| Service | Price (with watermark) | Price (no watermark) |
|---------|----------------------|---------------------|
| Dubbing v1 | $0.60/min | $0.90/min |

- Automatic speaker detection
- 29 languages
- MP3, MP4, WAV, MOV formats

---

## Price Variables (microcents)

```
# TTS - per 1K characters
TTS_FLASH_TURBO = 15_000_000      # $0.15 - eleven_flash_v2, eleven_flash_v2_5, eleven_turbo_v2, eleven_turbo_v2_5
TTS_MULTILINGUAL = 30_000_000     # $0.30 - eleven_multilingual_v2, eleven_v3

# Text-to-Dialogue - per 1K characters
DIALOGUE = 30_000_000             # $0.30 - eleven_v3 only

# STT - per hour
STT = 48_000_000                  # $0.48 - scribe_v1, scribe_v2

# Forced Alignment - per hour
FORCED_ALIGNMENT = 48_000_000     # $0.48

# Music - per minute
MUSIC = 48_000_000                # $0.48

# Voice Isolator - per minute
VOICE_ISOLATOR = 30_000_000       # $0.30

# Voice Changer - per minute
VOICE_CHANGER = 30_000_000        # $0.30 - eleven_multilingual_sts_v2, eleven_english_sts_v2

# Sound Effects - per generation
SOUND_EFFECTS = 18_000_000        # $0.18

# Dubbing - per minute
DUBBING_WATERMARK = 60_000_000    # $0.60
DUBBING_NO_WATERMARK = 90_000_000 # $0.90
```

## CEL Expression Examples

### TTS (per 1K characters, model-based pricing)
```cel
// outputs[0].extra.characters contains char count
// outputs[0].extra.model contains model name
(double(outputs[0].extra.characters) / 1000.0) * double(prices.per_1k_characters)
```

### STT (per hour, model in extra)
```cel
// inputs[0].seconds contains duration
// inputs[0].extra.model contains model name
(double(inputs[0].seconds) / 3600.0) * double(prices.per_hour)
```

### Audio Processing (per minute)
```cel
(double(inputs[0].seconds) / 60.0) * double(prices.per_minute)
```

### Sound Effects (flat fee per generation)
```cel
double(prices.per_generation)
```

### Text to Dialogue (per 1K total characters)
```cel
// outputs[0].extra.characters contains total char count
// outputs[0].extra.model contains model name
(double(outputs[0].extra.characters) / 1000.0) * double(prices.per_1k_characters)
```

### Voice Changer (per minute, model in extra)
```cel
// inputs[0].seconds contains duration
// inputs[0].extra.model contains model name
(double(inputs[0].seconds) / 60.0) * double(prices.per_minute)
```

### Forced Alignment (per hour of audio)
```cel
(double(inputs[0].seconds) / 3600.0) * double(prices.per_hour)
```
