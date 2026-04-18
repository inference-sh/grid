# Pricing: kling-video-v2-1-t2v

## fal.ai Base Price
- Endpoint: `fal-ai/kling-video/v2.1/master/text-to-video`
- Price: ~$0.14 per second of video (estimated)
- Currency: USD

## Price Variables (microcents)
- `per_second`: 14000000000 (= $0.14 * 100000000000, estimated)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * double(prices.per_second)
```

### pricing_description
```cel
"~$0.14 per second (estimated)"
```

## Calculation Notes
Kling Video v2.1 Master T2V — price estimated at $0.14/s based on comparable Kling models.
Update when official fal.ai pricing is published.
Duration options are 5 or 10 seconds.
