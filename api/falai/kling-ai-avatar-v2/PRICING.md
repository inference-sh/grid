# Pricing: kling-ai-avatar-v2

## fal.ai Base Price
- Standard endpoint: `fal-ai/kling-video/ai-avatar/v2/standard` — $0.0562 per second
- Pro endpoint: `fal-ai/kling-video/ai-avatar/v2/pro` — ~$0.112 per second (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_second_standard`: 5620000000 (= $0.0562 * 100000000000)
- `per_second_pro`: 11200000000 (= $0.112 * 100000000000) **estimated**

## CEL Expressions

### inference_expression
```cel
inputs[0].tier == "pro"
  ? double(outputs[0].seconds) * double(prices.per_second_pro)
  : double(outputs[0].seconds) * double(prices.per_second_standard)
```

### pricing_description
```cel
"$0.0562/s (standard), ~$0.112/s (pro, estimated)"
```

## Calculation Notes
Standard tier pricing confirmed at $0.0562/s.
Pro tier estimated at 2x standard ($0.112/s).
Update pro pricing when confirmed.
