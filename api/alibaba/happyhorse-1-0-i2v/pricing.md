# HappyHorse 1.0 Image-to-Video Pricing

## Model: happyhorse-1.0-i2v

Pricing is per-second of generated video. Resolution directly affects cost.

| Resolution | Price per second |
|------------|-----------------|
| 720P | $0.14 |
| 1080P | $0.24 |

## Billing Rules

- **Input images:** Free
- **Output videos:** Billed per second
- **Duration range:** 3-15 seconds
- **Failed calls:** No charge
- **Free quota:** New users get 10 seconds free (expires after ~3 months)

## Cost Factors

| Factor | Impact |
|--------|--------|
| Resolution | 1080P costs ~71% more than 720P |
| Duration | Longer videos cost more (billed per second) |
| Input image | No additional cost |

## Example

A 5-second video at 720P:
- Cost = $0.14 x 5 = **$0.70**

A 10-second video at 1080P:
- Cost = $0.24 x 10 = **$2.40**

## Rate Limits

- 300 RPM (requests per minute)

## References

- [Full pricing details](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- [Rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
