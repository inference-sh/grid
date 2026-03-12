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

### Video Generation (P-Video)

| Resolution | Draft Mode | Price per Second |
|------------|------------|------------------|
| 720p       | OFF        | $0.02/sec        |
| 720p       | ON         | $0.005/sec       |
| 1080p      | OFF        | $0.04/sec        |
| 1080p      | ON         | $0.01/sec        |

Rate Limit: 250 req/min

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

## Trainer Models (Not Implemented)

| Model               | Price               |
|---------------------|---------------------|
| P-Image Trainer     | $1.80 / 1000 steps  |
| P-Image Edit Trainer| $4.00 / 1000 steps  |

## Notes

- Inference Providers can benefit from preferential prices - contact Pruna
- WAN video pricing is per video (not per second)
- flux-klein-4b is extremely cheap at $0.0001/image
