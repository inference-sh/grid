# Wan 2.7 Video Edit Pricing

## Model: wan2.7-videoedit

Pricing is per-second of video. Resolution directly affects cost.

| Resolution | Price per second |
|------------|-----------------|
| 720P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |
| 1080P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |

## Billing Rules

- **Input videos:** Billed per second
- **Output videos:** Billed per second
- **Total billable duration:** input_video_duration + output_video_duration
- **Input video duration:** 2-10 seconds
- **Reference images:** Free
- **Failed calls:** No charge
- **Free quota:** New users may have a [free quota](https://www.alibabacloud.com/help/en/model-studio/new-free-quota)

## Cost Factors

| Factor | Impact |
|--------|--------|
| Resolution | 1080P costs more than 720P |
| Input video length | Input duration is billed |
| Output video length | Output duration is billed |
| Reference images | Free - no additional cost |
| Audio regeneration | No additional cost vs. keeping original |

## Billing Example

Edit a 5s video at 720P:
- Input video duration: 5s
- Output video duration: 5s
- **Total billed:** 10s at 720P rate

## Rate Limits

See [Wan series rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit#a729d7b6bar7y).

## References

- [Full pricing details](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- [Rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
