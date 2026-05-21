# Pricing: dia-tts

## fal.ai Base Price
- Endpoints: `fal-ai/dia-tts`, `fal-ai/dia-tts/voice-clone`
- Price: $0.04 per 1000 characters (both endpoints)
- Currency: USD

## Price Variables (microcents)
- `per_1k_characters`: 4000000 (= $0.04 * 100000000)

## CEL Expressions

### inference_expression
```cel
(double(size(inputs[0].text)) / 1000.0) * double(prices.per_1k_characters)
```

### pricing_description
```cel
"$0.04 per 1000 characters"
```

## Calculation Notes
fal.ai charges per 1000 characters of input text.
We measure input text length and apply the same rate.
