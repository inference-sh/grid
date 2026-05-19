# HeyGen Provider

AI video generation platform specializing in talking avatars, video translation, and lip-sync.

## Models

| App | Endpoint | Category | Description |
|-----|----------|----------|-------------|
| [avatar-video](avatar-video/) | `POST /v3/videos` (type=avatar) | video | Talking avatar videos with digital/photo avatars |
| [photo-video](photo-video/) | `POST /v3/videos` (type=image) | video | Animate portrait photos into talking videos |
| [video-agent](video-agent/) | `POST /v3/video-agents` | video | Prompt-to-video with AI agent (auto avatar/script) |
| [video-translate](video-translate/) | `POST /v3/video-translations` | video | Translate videos into 30+ languages with lip-sync |
| [lipsync](lipsync/) | `POST /v3/lipsyncs` | video | Re-sync lip movements to new audio |
| [text-to-speech](text-to-speech/) | `POST /v3/voices/speech` | audio | Text-to-speech with Starfish engine |

## Authentication

All apps use the `HEYGEN_API_KEY` secret. Get your API key from the [HeyGen dashboard](https://app.heygen.com/settings/api).

## API

- **Base URL:** `https://api.heygen.com`
- **Auth header:** `x-api-key`
- **API version:** v3
- **Async pattern:** POST to create -> poll GET for status -> download result

## Pricing

See [pricing.md](pricing.md) for full details.

| Feature | Price |
|---------|-------|
| Photo Avatar video (720p/1080p) | $0.05/sec |
| Photo Avatar video (4K) | $0.067/sec |
| Digital Twin video (720p/1080p) | $0.067/sec |
| Digital Twin video (4K) | $0.083/sec |
| Video Agent | $0.033/sec |
| Video Translation (audio-only) | $0.017/sec |
| Video Translation (lip-sync, speed) | $0.033/sec |
| Video Translation (lip-sync, precision) | $0.067/sec |
| Lipsync (speed) | $0.033/sec |
| Lipsync (precision) | $0.067/sec |
| Text-to-Speech | $0.00067/sec |

## References

- [HeyGen API Docs](https://developers.heygen.com/docs/quick-start)
- [API Reference](https://developers.heygen.com/reference)
