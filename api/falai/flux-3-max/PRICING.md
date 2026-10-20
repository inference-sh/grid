# Pricing: flux-3-max

## fal.ai Base Price
- Text-to-video: `blackforestlabs/flux-3-max/text-to-video`
- Image-to-video: `blackforestlabs/flux-3-max/image-to-video`
- Keyframes-to-video: `blackforestlabs/flux-3-max/keyframes-to-video`
- 720p: $0.17 per second of output video
- 1080p: $0.29 per second of output video
- Launch promo (75% off, Oct 15 to Nov 7, 2026): $0.0425/s at 720p, $0.0725/s at 1080p
- Currency: USD

## Price Variables (microcents, $1 = 100,000,000)
- `rate_720p_per_sec`: 17000000 (= $0.17)
- `rate_1080p_per_sec`: 29000000 (= $0.29)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * (outputs[0].resolution == "1080p"
  ? double(prices.rate_1080p_per_sec)
  : double(prices.rate_720p_per_sec))
```

### estimate (task_inputs.duration is an int 5-20, resolution "720p" | "1080p")
```cel
double(task_inputs.duration) * (task_inputs.resolution == "1080p"
  ? double(prices.rate_1080p_per_sec)
  : double(prices.rate_720p_per_sec))
```

### pricing_description
```cel
"$0.17 per second (720p), $0.29 per second (1080p)"
```

## Expected values
- 5s @ 720p: $0.85 = 85,000,000 microcents
- 5s @ 1080p: $1.45 = 145,000,000 microcents
- 20s @ 1080p: $5.80 = 580,000,000 microcents

## Calculation Notes
fal.ai charges per second of generated video. `outputs[0].seconds` is the requested duration (fal's
file can run a frame or two longer, e.g. 5.04s for a 5s request). Audio and reasoning level do not change the price.
