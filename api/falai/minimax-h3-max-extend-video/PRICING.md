# Pricing: minimax-h3-max-extend-video

## fal.ai Base Price
- `minimax/h3-max/extend-video`, billed per second of NEW footage (the requested duration)
- 480P $0.05/s, 768P $0.08/s, 1080P $0.16/s, 2K $0.32/s
- fal also charges $0.02 per 1k reference tokens beyond 4,096 included per request (source video). Not modelled.

## Price Variables (microcents)
- `rate_480p_per_sec`: 5000000, `rate_768p_per_sec`: 8000000, `rate_1080p_per_sec`: 16000000, `rate_2k_per_sec`: 32000000

## Expression
`outputs[0].seconds` = requested duration; `outputs[0].resolution` in {"480P","768P","1080P","2K"}.
Estimate exact from task_inputs.duration (default 5) and task_inputs.resolution (default "768P").

## Expected values
- 5s 768P: $0.40 = 40,000,000; 5s 2K: $1.60 = 160,000,000; 10s 480P: $0.50 = 50,000,000
