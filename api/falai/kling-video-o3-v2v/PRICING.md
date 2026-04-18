# Pricing: kling-video-o3-v2v

## fal.ai Base Price
- Pro edit: `fal-ai/kling-video/o3/pro/video-to-video/edit`
- Pro reference: `fal-ai/kling-video/o3/pro/video-to-video/reference`
- Standard edit: `fal-ai/kling-video/o3/standard/video-to-video/edit`
- Standard reference: `fal-ai/kling-video/o3/standard/video-to-video/reference`
- Price: ~$0.14 per second of video (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_second`: 14000000000 (= $0.14 * 100000000000, estimated)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### pricing_description
```cel
"~$0.14 per second (estimated)"
```

## Calculation Notes
Kling O3 V2V — no confirmed pricing. Estimated at $0.14/s.
Update when official fal.ai pricing is published.
