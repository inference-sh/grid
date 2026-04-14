# Pricing: patina

## fal.ai Base Price
- Endpoint: `fal-ai/patina`
- Price: $0.01 per megapixels
- Currency: USD

## Price Variables (microcents)
- `per_unit`: 1000000 (= $0.01 * 100000000)

## CEL Expressions

### inference_expression
```cel
double(prices.per_unit)
```

### pricing_description
```cel
"$0.01 per megapixels"
```

## Calculation Notes
fal.ai charges $0.01 per megapixels.
