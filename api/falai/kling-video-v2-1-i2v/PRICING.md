# Pricing: kling-video-v2-1-i2v

## fal.ai Base Price
- Master endpoint: `fal-ai/kling-video/v2.1/master/image-to-video` — $0.28 per second
- Pro endpoint: `fal-ai/kling-video/v2.1/pro/image-to-video` — ~$0.14 per second (estimated)
- Standard endpoint: `fal-ai/kling-video/v2.1/standard/image-to-video` — ~$0.07 per second (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_second`: 28000000000 (= $0.28 * 100000000000, master tier)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### pricing_description
```cel
"$0.28 per second (master)"
```

## Calculation Notes
Kling Video v2.1 I2V — master tier confirmed at $0.28/s.
Pro estimated at $0.14/s, standard estimated at $0.07/s.
Duration options are 5 or 10 seconds.
