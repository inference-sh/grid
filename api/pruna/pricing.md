# Pruna Pricing

Source: Pruna Developer Portal (2026-03)

## Image Models

| Model               | Price               | Rate Limit    |
|---------------------|---------------------|---------------|
| P-Image             | $0.005 / image      | 500 req/min   |
| P-Image-LoRA        | $0.005 / image      | 250 req/min   |
| P-Image-Edit        | $0.010 / image      | 500 req/min   |
| P-Image-Edit-LoRA   | $0.010 / image      | 250 req/min   |

## Video Model (P-Video)

| Resolution | Draft Mode | Price per Second |
|------------|------------|------------------|
| 720p       | OFF        | $0.02/sec        |
| 720p       | ON         | $0.005/sec       |
| 1080p      | OFF        | $0.04/sec        |
| 1080p      | ON         | $0.01/sec        |

Rate Limit: 250 req/min

Example: 5-second 720p video (draft off) = 5 × $0.02 = $0.10

## Trainer Models (Not Implemented)

| Model               | Price               |
|---------------------|---------------------|
| P-Image Trainer     | $1.80 / 1000 steps  |
| P-Image Edit Trainer| $4.00 / 1000 steps  |

## Notes

- Inference Providers can benefit from preferential prices - contact Pruna
- LoRA models have lower rate limits (250 vs 500 req/min)
- Video pricing varies significantly with resolution and draft mode
