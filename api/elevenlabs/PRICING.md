# ElevenLabs API Pricing

Based on Starter plan pricing (we charge at this tier).

## Text to Speech

| Model | Price per 1K chars |
|-------|-------------------|
| Flash / Turbo | $0.08 |
| Multilingual v2 / v3 | $0.17 |

- Flash: `eleven_flash_v2_5` - Ultra-low latency (~75ms), 32 languages
- Turbo: `eleven_turbo_v2_5` - Low latency
- Multilingual v2: `eleven_multilingual_v2` - Highest quality, ~250-300ms
- 40,000 character limit per request

## Speech to Text (Scribe)

| Model | Price per hour |
|-------|---------------|
| Scribe v1/v2 | $0.40 |
| Scribe v2 Realtime | $0.48 |

- 98%+ transcription accuracy
- 90+ languages
- Optional: keyterm prompting, entity detection, diarization
- Keyterm prompting: +$0.12/hour
- Entity detection: +$0.08/hour

## Music Generation

| Service | Price |
|---------|-------|
| Music compose | $0.35 per minute |

- Max 5 minute duration per request
- Commercial licensing included on Starter+
- 44.1kHz, 128-192kbps audio

## Voice Isolator

| Service | Price |
|---------|-------|
| Audio isolation | $0.22 per minute |

- Removes ambient sounds, reverb, interference
- WAV, MP3, FLAC, OGG, AAC inputs
- Max 500MB / 1 hour per file

## Voice Changer (Speech to Speech)

| Service | Price |
|---------|-------|
| Voice transformation | $0.22 per minute |

- Real-time processing
- 10,000+ human-like voices
- 70+ languages

## Sound Effects

| Service | Price |
|---------|-------|
| Sound generation | $0.13 per generation |

- Text-to-sound-effect
- Royalty-free
- MP3 (44.1kHz) or WAV (48kHz) output

## Text to Dialogue

| Model | Price per 1K chars |
|-------|-------------------|
| Flash / Turbo | $0.08 |
| Multilingual v2 | $0.17 |

- Same pricing as TTS (per character across all dialogue lines)
- Multi-speaker dialogue in single request
- Total character count = sum of all text fields in script

## Forced Alignment

| Service | Price per hour |
|---------|---------------|
| Forced alignment | $0.40 |

- Same pricing as STT
- Word-level and character-level timestamps
- Useful for subtitles, lip-sync, karaoke

## Dubbing

| Service | Price (with watermark) | Price (no watermark) |
|---------|----------------------|---------------------|
| Dubbing v1 | $0.44/min | $0.67/min |

- Automatic speaker detection
- 29 languages
- MP3, MP4, WAV, MOV formats

---

## Price Variables (microcents)

```
TTS Flash/Turbo: per_1k_characters = 8_000_000  ($0.08)
TTS Multilingual: per_1k_characters = 17_000_000 ($0.17)
Text-to-Dialogue: same as TTS (per 1K chars total)
STT: per_hour = 40_000_000 ($0.40)
Forced Alignment: per_hour = 40_000_000 ($0.40)
Music: per_minute = 35_000_000 ($0.35)
Voice Isolator: per_minute = 22_000_000 ($0.22)
Voice Changer: per_minute = 22_000_000 ($0.22)
Sound Effects: per_generation = 13_000_000 ($0.13)
Dubbing (watermark): per_minute = 44_000_000 ($0.44)
Dubbing (no watermark): per_minute = 67_000_000 ($0.67)
```

## CEL Expression Examples

### TTS (per 1K characters)
```cel
(double(size(inputs[0].text)) / 1000.0) * double(prices.per_1k_characters)
```

### STT (per hour)
```cel
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
(double(inputs[0].characters) / 1000.0) * double(prices.per_1k_characters)
```

### Forced Alignment (per hour of audio)
```cel
(double(inputs[0].seconds) / 3600.0) * double(prices.per_hour)
```
