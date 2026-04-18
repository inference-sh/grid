# Pricing: kling-video-v2-6-motion

## fal.ai Base Price
- Pro endpoint: `fal-ai/kling-video/v2.6/pro/motion-control` — $0.112 per second
- Standard endpoint: `fal-ai/kling-video/v2.6/standard/motion-control` — $0.07 per second
- Currency: USD

## Price Variables (microcents)
- `per_second_pro`: 11200000000 (= $0.112 * 100000000000)
- `per_second_standard`: 7000000000 (= $0.07 * 100000000000)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * (outputs[0].extra.tier == "pro" ? double(prices.per_second_pro) : double(prices.per_second_standard))
```

### pricing_description
```cel
"$0.112/s (pro) or $0.07/s (standard)"
```

## Calculation Notes
Kling Video v2.6 Motion Control has two tiers with different per-second pricing.
The tier is stored in the output metadata extra field.
