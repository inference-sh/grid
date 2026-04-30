# HappyHorse 1.0 Video Edit Pricing

## Model: happyhorse-1.0-video-edit

Pricing is per-second of video. Resolution directly affects cost.

| Resolution | Price per second |
|------------|-----------------|
| 720P | $0.14 |
| 1080P | $0.24 |

## Billing Rules

- **Input videos:** Billed per second
- **Output videos:** Billed per second
- **Total billable duration:** input_video_duration + output_video_duration
- **Input video range:** 3-60 seconds (output capped at 15 seconds)
- **Reference images:** Free (up to 5)
- **Failed calls:** No charge
- **Free quota:** New users get 10 seconds free (expires after ~3 months)

## Cost Factors

| Factor | Impact |
|--------|--------|
| Resolution | 1080P costs ~71% more than 720P |
| Input video duration | Input video seconds are also billed |
| Output video duration | Output seconds are billed |
| Number of reference images | No additional cost (0-5 images) |

## Example

Editing a 6-second video at 720P (output = 6 seconds):
- Billable duration = 6s input + 6s output = 12s
- Cost = $0.14 x 12 = **$1.68**

If input video is longer than 15 seconds, only the first 15 seconds are used:
- Max output duration = 15 seconds

## Rate Limits

- 300 RPM (requests per minute)

## References

- [Full pricing details](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- [Rate limits](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
