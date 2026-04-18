# Pricing: kling-create-voice

## fal.ai Base Price
- Endpoint: `fal-ai/kling-video/create-voice`
- Price: ~$0.02 per request (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_request`: 2000000000 (= $0.02 * 100000000000) **estimated**

## CEL Expressions

### inference_expression
```cel
double(prices.per_request)
```

### pricing_description
```cel
"~$0.02 per request (estimated)"
```

## Calculation Notes
No confirmed pricing from fal.ai. Estimated at $0.02 per request.
Update when official pricing is available.
