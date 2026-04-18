# Pricing: kling-video-v2-6-t2v

## fal.ai Base Price
- Endpoint: `fal-ai/kling-video/v2.6/pro/text-to-video`
- Price: $0.07 per second of video
- Currency: USD

## Price Variables (microcents)
- `per_second`: 7000000000 (= $0.07 * 100000000000)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### pricing_description
```cel
"$0.07 per second"
```

## Calculation Notes
Kling Video v2.6 Pro charges per second of generated video.
Duration options are 5 or 10 seconds, so total cost is $0.35 or $0.70.
