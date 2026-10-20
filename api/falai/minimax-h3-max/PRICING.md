# Pricing: minimax-h3-max

## fal.ai Base Price
- Text-to-video: `minimax/h3-max/text-to-video`
- Image-to-video: `minimax/h3-max/image-to-video`
- Reference-to-video: `minimax/h3-max/reference-to-video`
- 480P: $0.05 per second
- 768P: $0.08 per second
- 1080P: $0.16 per second
- Promo (35% off, ends Oct 31, 2026): $0.0325 / $0.052 / $0.104 per second
- Currency: USD

## Price Variables (microcents, $1 = 100,000,000)
- `rate_480p_per_sec`: 5000000 (= $0.05)
- `rate_768p_per_sec`: 8000000 (= $0.08)
- `rate_1080p_per_sec`: 16000000 (= $0.16)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].seconds) * (outputs[0].resolution == "1080P"
  ? double(prices.rate_1080p_per_sec)
  : (outputs[0].resolution == "480P"
      ? double(prices.rate_480p_per_sec)
      : double(prices.rate_768p_per_sec)))
```

### estimate (task_inputs.duration 1-15, resolution "480P" | "768P" | "1080P")
```cel
double(task_inputs.duration) * (task_inputs.resolution == "1080P"
  ? double(prices.rate_1080p_per_sec)
  : (task_inputs.resolution == "480P"
      ? double(prices.rate_480p_per_sec)
      : double(prices.rate_768p_per_sec)))
```

### pricing_description
```cel
"$0.05 per second (480P), $0.08 per second (768P), $0.16 per second (1080P)"
```

## Expected values
- 5s @ 480P: $0.25 = 25,000,000 microcents
- 5s @ 768P: $0.40 = 40,000,000 microcents
- 5s @ 1080P: $0.80 = 80,000,000 microcents
- 15s @ 1080P: $2.40 = 240,000,000 microcents

## Calculation Notes
fal.ai charges per second of generated video at the chosen resolution. `outputs[0].seconds` is the requested
duration (the output can run up to ~0.7s longer). Prompt expansion and target audio do not change the price.
