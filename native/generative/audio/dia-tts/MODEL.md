# Model: Dia TTS (Consolidated)

Consolidates two fal.ai endpoints:
- `fal-ai/dia-tts` - Basic text-to-speech
- `fal-ai/dia-tts/voice-clone` - Voice cloning from reference audio

## Endpoints

| Mode | Endpoint | Trigger |
|------|----------|---------|
| Basic TTS | `fal-ai/dia-tts` | No ref_audio provided |
| Voice Clone | `fal-ai/dia-tts/voice-clone` | ref_audio provided |

## Category
text-to-speech / audio-to-audio

## Description
Dia generates realistic dialogue from transcripts with emotion control and natural nonverbals (laughter, throat clearing). Voice cloning mode allows matching a reference voice.

## Input Schema

### Required Fields
- `text` (string): The text to convert to speech

### Optional Fields (Voice Cloning)
- `ref_audio` (File): Reference audio file for voice cloning
- `ref_text` (string): Transcript of the reference audio (required if ref_audio provided)

### Text Format
Use speaker tags `[S1]`, `[S2]` for multi-speaker dialogue and parenthetical expressions for nonverbals:

```
[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Fal.
```

## Output Schema
- `audio` (File): The generated speech audio

## Modes

### Basic TTS
Just provide `text`. Uses default Dia voices.

```json
{
  "text": "[S1] Hello! [S2] Hi there!"
}
```

### Voice Clone
Provide `text` + `ref_audio` + `ref_text`. Clones voice from reference.

```json
{
  "text": "[S1] Hello! [S2] Hi there!",
  "ref_audio": {"uri": "https://example.com/voice-sample.mp3"},
  "ref_text": "[S1] This is my voice sample. [S2] And this is another speaker."
}
```

## Notes
- Same pricing for both modes ($0.04 per 1000 characters)
- ref_text must match the speaker tags used in ref_audio
- Output is .wav format for voice-clone, .mp3 for basic TTS
