# Pricing: minimax-h3-max-relight

## fal.ai Base Price
- `minimax/h3-max/relight`, per second of output video (= prepared source length, up to 15s)
- 480P $0.05/s, 768P $0.08/s, 1080P $0.16/s, 2K $0.32/s

## Price Variables (microcents)
- `rate_480p_per_sec`: 5000000, `rate_768p_per_sec`: 8000000, `rate_1080p_per_sec`: 16000000, `rate_2k_per_sec`: 32000000

## Expression
`outputs[0].seconds` is probed from the output file with ffprobe. Estimate is a range (source length unknown, max 15s).

## Expected values
- 5s 768P: $0.40 = 40,000,000; 15s 2K: $4.80 = 480,000,000
