# Model: Kokoro TTS (Consolidated)

Consolidates 9 fal.ai Kokoro endpoints into a single app with language selection.

## Endpoints

| Language | Endpoint | Default Voice |
|----------|----------|---------------|
| American English | `fal-ai/kokoro/american-english` | af_heart |
| British English | `fal-ai/kokoro/british-english` | bf_alice |
| French | `fal-ai/kokoro/french` | ff_siwis |
| Spanish | `fal-ai/kokoro/spanish` | ef_dora |
| Japanese | `fal-ai/kokoro/japanese` | jf_alpha |
| Italian | `fal-ai/kokoro/italian` | if_sara |
| Hindi | `fal-ai/kokoro/hindi` | hf_alpha |
| Brazilian Portuguese | `fal-ai/kokoro/brazilian-portuguese` | pf_dora |
| Mandarin Chinese | `fal-ai/kokoro/mandarin-chinese` | zf_xiaobei |

## Category
text-to-audio

## Description
Kokoro is a lightweight text-to-speech model that delivers comparable quality to larger models while being significantly faster and more cost-efficient. Supports 9 languages with multiple voice options.

## Input Schema

### Required Fields
- `prompt` (string): Text to convert to speech

### Optional Fields
- `language` (Literal, default: american-english): Language for speech generation
- `voice` (string, default: per-language default): Voice ID
- `speed` (float, default: 1.0): Speed of the generated audio (0.1-5.0)

## Voices by Language

### American English
af_heart, af_alloy, af_aoede, af_bella, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa

### British English
bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis

### French
ff_siwis

### Spanish
ef_dora, em_alex, em_santa

### Japanese
jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo

### Italian
if_sara, im_nicola

### Hindi
hf_alpha, hf_beta, hm_omega, hm_psi

### Brazilian Portuguese
pf_dora, pm_alex, pm_santa

### Mandarin Chinese
zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang

## Output Schema
- `audio` (File): The generated speech audio (.wav)

## Notes
- Voice IDs use prefix convention: first letter = language, second = gender (f=female, m=male)
- Same pricing across all languages ($0.02 per 1000 characters)
- Invalid voice for a language raises a validation error with available options
