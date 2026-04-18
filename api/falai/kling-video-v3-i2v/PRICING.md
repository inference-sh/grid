# Pricing: kling-video-v3-i2v

## fal.ai Base Price
- Endpoints: `fal-ai/kling-video/v3/pro/image-to-video`, `fal-ai/kling-video/v3/standard/image-to-video`
- Price: $0.14 per second of video
- Currency: USD

## Price Variables (microcents)
- `per_second`: 14000000000 (= $0.14 * 100000000000)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### pricing_description
```cel
"$0.14 per second of video"
```

## Calculation Notes
fal.ai charges $0.14 per second of generated video. Both pro and standard tiers
are priced the same. Duration ranges from 3 to 15 seconds.
