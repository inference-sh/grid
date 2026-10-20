# Pricing: minimax-h3-max-turbo-extend-video

## fal.ai Base Price
- `minimax/h3-max-turbo/extend-video`, billed per second of NEW footage (the requested duration)
- 480P $0.025/s, 768P $0.04/s, 1080P $0.08/s, 2K $0.16/s

## Price Variables (microcents)
- `rate_480p_per_sec`: 2500000, `rate_768p_per_sec`: 4000000, `rate_1080p_per_sec`: 8000000, `rate_2k_per_sec`: 16000000

## Expression
Same shape as minimax-h3-max-extend-video. Estimate exact from task_inputs.duration (default 5) and resolution (default "768P").

## Expected values
- 5s 768P: $0.20 = 20,000,000; 5s 2K: $0.80 = 80,000,000; 10s 480P: $0.25 = 25,000,000
