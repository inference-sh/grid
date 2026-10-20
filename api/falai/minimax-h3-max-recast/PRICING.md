# Pricing: minimax-h3-max-recast

## fal.ai Base Price
- `minimax/h3-max/recast`, per second of output video (= source length, 5-30s). Reference images are free.
- 768P $0.30/s, 1080P $0.45/s

## Price Variables (microcents)
- `rate_768p_per_sec`: 30000000, `rate_1080p_per_sec`: 45000000

## Expression
`outputs[0].seconds` is probed from the output file. Estimate is a range (5-30s source).

## Expected values
- 5s 1080P: $2.25 = 225,000,000; 10s 768P: $3.00 = 300,000,000; 30s 1080P: $13.50 = 1,350,000,000
