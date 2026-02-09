# Pricing: kokoro-tts

## fal.ai Base Price
- Endpoints: `fal-ai/kokoro/*` (all languages)
- Price: $0.02 per 1000 characters
- Currency: USD

## Price Variables (microcents)
- `per_1k_characters`: 2000000 (= $0.02 * 100000000)

## CEL Expressions

### inference_expression
```cel
(double(size(inputs[0].text)) / 1000.0) * double(prices.per_1k_characters)
```

### pricing_description
```cel
"$0.02 per 1000 characters"
```

## Calculation Notes
fal.ai charges $0.02 per 1000 characters of input text. Same rate for all languages.
