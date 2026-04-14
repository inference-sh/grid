# Pricing: extract

## fal.ai Base Price
- Endpoint: `fal-ai/patina/material/extract`
- Price: $7e-05 per compute seconds
- Currency: USD

## Price Variables (microcents)
- `per_unit`: 6999 (= $7e-05 * 100000000)

## CEL Expressions

### inference_expression
```cel
double(prices.per_unit)
```

### pricing_description
```cel
"$7e-05 per compute seconds"
```

## Calculation Notes
fal.ai charges $7e-05 per compute seconds.
