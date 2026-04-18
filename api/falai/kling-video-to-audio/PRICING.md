# Pricing: kling-video-to-audio

## fal.ai Base Price
- Endpoint: `fal-ai/kling-video/video-to-audio`
- Price: ~$0.14 per request (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_request`: 14000000000 (= $0.14 * 100000000000) **estimated**

## CEL Expressions

### inference_expression
```cel
double(prices.per_request)
```

### pricing_description
```cel
"~$0.14 per request (estimated)"
```

## Calculation Notes
No confirmed pricing from fal.ai. Estimated at $0.14 per request.
Update when official pricing is available.
