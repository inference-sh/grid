# Pruna Pricing

Source: Pruna Developer Portal (2026-03)

## P-Series Models

### Image Generation

| Model               | Price               | Rate Limit    |
|---------------------|---------------------|---------------|
| P-Image             | $0.005 / image      | 500 req/min   |
| P-Image-LoRA        | $0.005 / image      | 250 req/min   |
| P-Image-Edit        | $0.010 / image      | 500 req/min   |
| P-Image-Edit-LoRA   | $0.010 / image      | 250 req/min   |

### P-Image-Ideogram

Priced per image by thinking level and resolution budget. 2K is 2x the 1K rate.

| thinking   | 1K       | 2K      |
|------------|----------|---------|
| very low   | $0.003   | $0.006  |
| low        | $0.0075  | $0.015  |
| medium     | $0.010   | $0.020  |
| high       | $0.015   | $0.030  |

Rate Limit: 500 req/min

### Video Generation (P-Video)

| Resolution | Draft Mode | Price per Second |
|------------|------------|------------------|
| 720p       | OFF        | $0.02/sec        |
| 720p       | ON         | $0.005/sec       |
| 1080p      | OFF        | $0.04/sec        |
| 1080p      | ON         | $0.01/sec        |

Rate Limit: 250 req/min

Microcents below: $1 = 100,000,000. Billed at list price; partners pay 15% off list.

### P-Video-2 (per second of the finished video, whole seconds)

| Resolution | draft=false | draft=true |
|------------|-------------|------------|
| 720p       | $0.025 (2,500,000) | $0.015 (1,500,000) |
| 1080p      | $0.05 (5,000,000)  | $0.03 (3,000,000)  |

Rate Limit: 250 req/min

### P-Video-2-Pro (per second, whole seconds = int(frames/24))

| Resolution | cost | speed | quality |
|------------|------|-------|---------|
| 480p  | $0.01  | $0.02  | $0.04  |
| 768p  | $0.025 | $0.035 | $0.075 |
| 1080p | $0.045 | $0.055 | $0.095 |
| 2k    | $0.065 | $0.075 | $0.115 |
| 4k    | $0.105 | $0.115 | $0.155 |

1080p/2k/4k = 768p rate + p-video-upscale rate. Rate Limit: 250 req/min

### P-Video-2-Pro-Reference

| Output /s | cost | speed |
|-----------|------|-------|
| 480p  | $0.015 | $0.02  |
| 768p  | $0.025 | $0.035 |
| 1080p | $0.045 | $0.055 |
| 2k    | $0.065 | $0.075 |
| 4k    | $0.105 | $0.115 |

Reference video /s: cost 480p $0.03, 768p+ $0.06; speed 480p $0.04, 768p+ $0.06.
Reference image: cost $0.005, speed $0.01 each (first/last frames count).
Reference audio /s: cost $0.0015, speed $0.0025.
Each reference clip bills up to output + 0.5s; reference video capped at 15s combined.
Sample: 480p cost, 5s out, 1 image, 5.167s video, 3.9s audio = 24,086,000.

### P-Video-2-Pro-Avatar (per second of output)

480p $0.02 · 768p $0.04 · 1080p $0.06 · 2k $0.08 · 4k $0.12. Rate Limit: 50 req/min

### P-Video-Upscale (per second of output, rounded up)

| Target | <=30 fps | >30 fps |
|--------|----------|---------|
| 720p   | $0.01 | $0.02 |
| 1080p  | $0.02 | $0.04 |
| 2k     | $0.04 | $0.08 |
| 4k     | $0.08 | $0.16 |

Processed-frame cap: 720p 2400, 1080p 1200, 2k 600, 4k 300. Rate Limit: 50 req/min

### P-Video-Animate (per second of output)

720p $0.03 · 1080p $0.06. Rate Limit: 50 req/min

### VACE

$0.40 per video (40,000,000). Rate Limit: 30 req/min

### P-Image-Try-On

15 units for the first image input + 8 per additional (person photo, garments, pose).
Unit value not stated by Pruna; unpriced until confirmed. Rate Limit: 500 req/min

## FLUX Models

| Model          | Price          | Rate Limit    |
|----------------|----------------|---------------|
| flux-dev       | $0.005 / image | 150 req/min   |
| flux-dev-lora  | $0.01 / image  | 150 req/min   |
| flux-klein-4b  | $0.0001 / image| 150 req/min   |

## Qwen Models

| Model               | Price          | Rate Limit    |
|---------------------|----------------|---------------|
| qwen-image          | $0.025 / image | 150 req/min   |
| qwen-image-fast     | $0.005 / image | 150 req/min   |
| qwen-image-edit-plus| $0.03 / image  | 150 req/min   |

## Z-Image Models

| Model             | Price          | Rate Limit    |
|-------------------|----------------|---------------|
| z-image-turbo     | $0.005 / image | 150 req/min   |
| z-image-turbo-lora| $0.008 / image | 150 req/min   |

## WAN Models

| Model           | Price          | Rate Limit    |
|-----------------|----------------|---------------|
| wan-image-small | $0.005 / image | 150 req/min   |

### wan-t2v (text-to-video)

| Resolution | Price / video |
|------------|---------------|
| 480p       | $0.05         |
| 720p       | $0.10         |

Rate Limit: 30 req/min

### wan-i2v (image-to-video)

| Resolution | Price / video |
|------------|---------------|
| 480p       | $0.05         |
| 720p       | $0.11         |

Rate Limit: 30 req/min

## Trainer Models

| Model               | Price               | Rate Limit |
|---------------------|---------------------|------------|
| P-Image Trainer     | $1.80 / 1000 steps  | 5/min      |
| P-Image Edit Trainer| $4.00 / 1000 steps  | 5/min      |

## Notes

- Inference Providers can benefit from preferential prices - contact Pruna
- WAN video pricing is per video (not per second)
- flux-klein-4b is extremely cheap at $0.0001/image
