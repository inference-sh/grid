# Pricing: minimax-h3-max-camera-controls

## fal.ai Base Price
- `minimax/h3-max/camera-controls`. fal's page shows no rate table; assumed to match H3 Max generation:
  480P $0.05/s, 768P $0.08/s, 1080P $0.16/s. Verify against the fal invoice after the first runs.

## Price Variables (microcents)
- `rate_480p_per_sec`: 5000000, `rate_768p_per_sec`: 8000000, `rate_1080p_per_sec`: 16000000

## Expression
`outputs[0].seconds` = requested duration; `outputs[0].resolution` in {"480P","768P","1080P"}; resolution default is "480P".

## Expected values
- 5s 480P (default): $0.25 = 25,000,000; 5s 768P: $0.40 = 40,000,000
