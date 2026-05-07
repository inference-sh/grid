# Pricing: v2

## fal.ai Base Price
- Endpoint: `fal-ai/hyper3d/rodin/v2`
- Price: $0.4 per generations
- Currency: USD

## Price Variables (microcents)
- `per_unit`: 40000000 (= $0.4 * 100000000)

## CEL Expressions

### inference_expression
```cel
double(prices.per_unit)
```

### pricing_description
```cel
"$0.4 per generations"
```

## Calculation Notes
fal.ai charges $0.4 per generations.
