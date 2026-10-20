# Pricing: minimax-h3-max-insert-video

## fal.ai Base Price
- `minimax/h3-max/insert-video`, per second of the newly generated scene only
- 480p $0.05/s, 768p $0.06/s
- $0.02 per 1k reference tokens beyond 4,096 included. Not modelled.

## Price Variables (microcents)
- `rate_480p_per_sec`: 5000000, `rate_768p_per_sec`: 6000000

## Expression
`outputs[0].seconds` = fal's injected_duration (falls back to the requested duration); `outputs[0].resolution` in {"480p","768p"} (lowercase).
Estimate exact from task_inputs.duration (default 5) and resolution (default "768p").

## Expected values
- 5s 768p: $0.30 = 30,000,000; 13s 480p: $0.65 = 65,000,000
