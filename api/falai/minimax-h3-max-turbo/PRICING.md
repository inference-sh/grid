# Pricing: minimax-h3-max-turbo

## fal.ai Base Price
- `minimax/h3-max-turbo/text-to-video`, `minimax/h3-max-turbo/image-to-video`
- 480P $0.025/s, 768P $0.04/s, 1080P $0.08/s (list; 35% launch promo until Oct 31, 2026)

## Price Variables (microcents, $1 = 100,000,000)
- `rate_480p_per_sec`: 2500000
- `rate_768p_per_sec`: 4000000
- `rate_1080p_per_sec`: 8000000

## Expression
`outputs[0].seconds` = requested duration; `outputs[0].resolution` in {"480P","768P","1080P"}. Same structure as minimax-h3-max.
Estimate: `double(task_inputs.duration) * rate(task_inputs.resolution)`, default 5s 768P = 20000000 ($0.20).

## Expected values
- 5s 768P: $0.20 = 20,000,000
- 5s 1080P: $0.40 = 40,000,000
- 15s 480P: $0.375 = 37,500,000
