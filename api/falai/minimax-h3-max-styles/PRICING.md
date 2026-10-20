# Pricing: minimax-h3-max-styles

## fal.ai Base Price
- `minimax/h3-max/styles/{vhs,hand-drawn,16bit-pixel,retro-toon-70s,low-poly}`: $0.08 per second, 768P only, all five styles.

## Price Variables (microcents)
- `rate_per_sec`: 8000000

## Expression
`outputs[0].seconds` = requested duration (int 5-15). Estimate exact: `double(task_inputs.duration) * rate_per_sec`, default 5s = 40000000.

## Expected values
- 5s: $0.40 = 40,000,000; 15s: $1.20 = 120,000,000
