# Pricing: kling-video-v2-5-turbo-i2v

## fal.ai Base Price
- Pro endpoint: `fal-ai/kling-video/v2.5-turbo/pro/image-to-video` — $0.07 per second
- Standard endpoint: `fal-ai/kling-video/v2.5-turbo/standard/image-to-video` — ~$0.035 per second (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_second`: 7000000000 (= $0.07 * 100000000000, pro tier)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### pricing_description
```cel
"$0.07 per second (pro)"
```

## Calculation Notes
Kling Video v2.5 Turbo I2V charges per second of generated video.
Pro tier is $0.07/s. Standard tier pricing estimated at ~$0.035/s.
Duration options are 5 or 10 seconds.
