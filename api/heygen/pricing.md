# HeyGen Pricing

All pricing is per-second of output duration. Billed against prepaid USD wallet balance.

## Avatar Video (avatar-video, photo-video)

| Avatar Type | Resolution | Price per second |
|-------------|-----------|-----------------|
| Photo Avatar | 720p / 1080p | $0.05 |
| Photo Avatar | 4K | $0.067 |
| Digital Twin | 720p / 1080p | $0.067 |
| Digital Twin | 4K | $0.083 |
| Studio Avatar | 720p / 1080p | $0.067 |
| Studio Avatar | 4K | $0.083 |

## Video Agent (video-agent)

| Feature | Price per second |
|---------|-----------------|
| Prompt-to-video | $0.033 |

## Video Translation (video-translate)

| Mode | Price per second |
|------|-----------------|
| Audio-only | $0.017 |
| Lip-sync (speed) | $0.033 |
| Lip-sync (precision) | $0.067 |

## Lipsync (lipsync)

| Mode | Price per second |
|------|-----------------|
| Speed | $0.033 |
| Precision | $0.067 |

## Text-to-Speech (text-to-speech)

| Engine | Price per second |
|--------|-----------------|
| Starfish | $0.00067 |

## Avatar Creation (not included in grid apps)

| Type | Price per call |
|------|---------------|
| Digital Twin | $1.00 |
| Photo Avatar | $1.00 |

## Billing Rules

- **Billed by duration:** Cost = unit price x output duration (seconds)
- **Failed calls:** No charge
- **Billing model:** Prepaid USD wallet, consumed per API call

## Examples

A 30-second avatar video at 1080p (photo avatar):
- Cost = $0.05 x 30 = **$1.50**

A 60-second video translated with lip-sync (speed):
- Cost = $0.033 x 60 = **$1.98**

A 10-second lipsync (precision):
- Cost = $0.067 x 10 = **$0.67**

60 seconds of TTS audio:
- Cost = $0.00067 x 60 = **$0.04**

## References

- [HeyGen Pricing](https://developers.heygen.com/docs/pricing)
