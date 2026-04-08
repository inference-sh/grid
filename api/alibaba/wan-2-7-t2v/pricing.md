# Wan 2.7 Text-to-Video Pricing

## Model: wan2.7-t2v

Pricing is per-second of generated video. Resolution directly affects cost.

| Resolution | Price per second |
|------------|-----------------|
| 720P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |
| 1080P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |

## Billing Rules

- **Billed by duration:** Cost = Unit price (based on resolution) x Duration (seconds)
- **Duration range:** 2-15 seconds
- **Failed calls:** No charge
- **Free quota:** New users may have a [free quota](https://www.alibabacloud.com/help/en/model-studio/new-free-quota)

## Cost Factors

| Factor | Impact |
|--------|--------|
| Resolution | 1080P costs more than 720P |
| Duration | Longer videos cost more (billed per second) |
| Prompt extend | No additional cost (increases processing time only) |

## Rate Limits

See [Wan series rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit#a729d7b6bar7y).

## References

- [Full pricing details](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- [Rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
