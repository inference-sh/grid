# Wan 2.7 Reference-to-Video Pricing

## Model: wan2.7-r2v

Pricing is per-second of video. Resolution directly affects cost.

| Resolution | Price per second |
|------------|-----------------|
| 720P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |
| 1080P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |

## Billing Rules

- **Input images:** Free
- **Input reference videos:** Billed per second
- **Output videos:** Billed per second
- **Total billable duration:** input_video_duration + output_video_duration
- **Duration range:** 2-10 (with ref video) or 2-15 (without)
- **Failed calls:** No charge
- **Free quota:** New users may have a [free quota](https://www.alibabacloud.com/help/en/model-studio/new-free-quota)

## Cost Factors

| Factor | Impact |
|--------|--------|
| Resolution | 1080P costs more than 720P |
| Duration | Longer videos cost more (billed per second) |
| Reference videos | Input video duration is also billed |
| Reference images | Free - no additional cost |
| Number of characters | No per-character cost, but more complex scenes may need longer duration |

## Billing Example

With 2 reference videos (5s each) and output duration of 10s:
- Input video duration: 5s + 5s = 10s (depends on actual usage reported)
- Output video duration: 10s
- **Total billed:** input_video_duration + output_video_duration

## Rate Limits

See [Wan series rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit#a729d7b6bar7y).

## References

- [Full pricing details](https://www.alibabacloud.com/help/en/model-studio/model-pricing#5c3d28ad8a4x8)
- [Rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
