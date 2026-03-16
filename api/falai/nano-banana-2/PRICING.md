# Pricing: nano-banana-2

## fal.ai Base Price
- Endpoint: `fal-ai/nano-banana-2` and `fal-ai/nano-banana-2/edit`
- Price: $0.08 per image
- Currency: USD

## Price Variables (microcents)
- `per_image`: 8000000000 (calculation: $0.08 * 100000000)

## CEL Expressions

### inference_expression
```cel
output_meta.outputs.sum(o, o.count) * 8000000000
```

### pricing_description
```cel
"$0.08 per image"
```

## Calculation Notes
- Same price for both text-to-image and edit modes
- Price scales with number of images generated (num_images parameter)
- Microcents = cents * 1,000,000 = dollars * 100,000,000
