# Pricing: minimax-h3-max-lip-sync

## fal.ai Base Price
- `minimax/h3-max/lip-sync/image-to-video`, per second of generated video (= audio length)
- 480P $0.05/s, 768P $0.08/s, 1080P $0.16/s, 2K $0.32/s
- Videos longer than 15 seconds are billed at 1.2x (fal: 30s at 768P = $2.88).

## Price Variables (microcents)
- `rate_480p_per_sec`: 5000000, `rate_768p_per_sec`: 8000000, `rate_1080p_per_sec`: 16000000, `rate_2k_per_sec`: 32000000
- `long_form_multiplier`: 1.2 (apply when outputs[0].seconds > 15)

## Expression
`outputs[0].seconds` = fal's reported output duration (the audio length). Multiply by 1.2 when seconds > 15.
Estimate is a range: the audio length is not known before the run (5s to 900s).

## Expected values
- 5s 768P: $0.40 = 40,000,000; 30s 768P: $2.88 = 288,000,000; 10s 2K: $3.20 = 320,000,000
