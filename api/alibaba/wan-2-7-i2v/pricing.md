# Wan 2.7 Image-to-Video Pricing

## Model: wan2.7-i2v

Pricing is per-second of video. Resolution directly affects cost.

| Resolution | Price per second |
|------------|-----------------|
| 720P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |
| 1080P | See [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing) |

## Billing Rules

- **Input images:** Free
- **Input videos (continuation):** Billed per second
- **Output videos:** Billed per second
- **Total billable duration:** input_video_duration + output_video_duration
- **Duration range:** 2-15 seconds
- **Failed calls:** No charge
- **Free quota:** New users may have a [free quota](https://www.alibabacloud.com/help/en/model-studio/new-free-quota)

## Cost Factors

| Factor | Impact |
|--------|--------|
| Resolution | 1080P costs more than 720P |
| Duration | Longer videos cost more (billed per second) |
| Video continuation | Input video duration is also billed |
| Audio input | No additional cost for driving audio |

## Video Continuation Billing Example

If `duration=15` and input clip is 3s:
- Model generates 12s of new content
- Output video is 15s total
- **Billed for 15s** (3s input + 12s output)

## Rate Limits

See [Wan series rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit#a729d7b6bar7y).

## References

- [Full pricing details](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- [Rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
