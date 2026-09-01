# Pricing: minimax-h3-max

## fal.ai Base Price
- Text-to-video: `fal-ai/minimax/h3-max/text-to-video`
- Image-to-video: `fal-ai/minimax/h3-max/image-to-video`
- Reference-to-video: `fal-ai/minimax/h3-max/reference-to-video`
- 480P: $0.05 per second
- 768P: $0.08 per second
- Currency: USD

## Price Variables (microcents)
- `per_second_480p`: 5000000000 (= $0.05 * 100000000000)
- `per_second_768p`: 8000000000 (= $0.08 * 100000000000)

## CEL Expressions

### inference_expression
```cel
(outputs[0].extra.resolution == "480P")
  ? double(outputs[0].seconds) * double(prices.per_second_480p)
  : double(outputs[0].seconds) * double(prices.per_second_768p)
```

### pricing_description
```cel
"$0.08 per second (768P), $0.05 per second (480P)"
```

## Calculation Notes
fal.ai charges per second of generated video. 768P costs $0.08/s, 480P costs $0.05/s.
Duration ranges from 5-15 seconds, so total cost at 768P ranges from $0.40 to $1.20.
