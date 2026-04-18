# Pricing: kling-video-v3-t2v

## fal.ai Base Price
- Pro endpoint: `fal-ai/kling-video/v3/pro/text-to-video`
- Standard endpoint: `fal-ai/kling-video/v3/standard/text-to-video`
- Price: $0.14 per second (both tiers)
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
"$0.14 per second"
```

## Calculation Notes
fal.ai charges $0.14 per second of generated video, same price for both pro and standard tiers.
Duration ranges from 3-15 seconds, so total cost ranges from $0.42 to $2.10 per generation.
