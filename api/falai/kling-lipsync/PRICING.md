# Pricing: kling-lipsync

## fal.ai Base Price
- Audio endpoint: `fal-ai/kling-video/lipsync/audio-to-video`
- Text endpoint: `fal-ai/kling-video/lipsync/text-to-video`
- Price: ~$0.05 per request (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_request`: 5000000000 (= $0.05 * 100000000000) **estimated**

## CEL Expressions

### inference_expression
```cel
double(prices.per_request)
```

### pricing_description
```cel
"~$0.05 per request (estimated)"
```

## Calculation Notes
No confirmed pricing from fal.ai. Estimated at $0.05 per request.
Update when official pricing is available.
