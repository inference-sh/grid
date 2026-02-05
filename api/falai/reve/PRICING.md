# Pricing: reve

## fal.ai Base Price
- Endpoints: `fal-ai/reve/text-to-image`, `fal-ai/reve/edit`, `fal-ai/reve/remix`
- Price: $0.04 per image
- Currency: USD

## Price Variables (microcents)
- `per_image`: 4000000 (= $0.04 * 100000000)

## CEL Expressions

### inference_expression
```cel
double(outputs[0].count) * double(prices.per_image)
```

### pricing_description
```cel
"$0.04 per image"
```

## Calculation Notes
fal.ai charges $0.04 per generated image, same for all modes (text-to-image, edit, remix).
