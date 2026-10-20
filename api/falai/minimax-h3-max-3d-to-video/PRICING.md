# Pricing: minimax-h3-max-3d-to-video

## fal.ai Base Price
- `minimax/h3-max/3d-to-video`
- $0.50 processing fee per request
- 480P $0.05/s, 768P $0.08/s, 1080P $0.16/s per billed second; each shot rounded up to whole seconds, 5s minimum per shot
- $0.10 per automatically generated reference image (only when no reference images are supplied)
- $0.02 per 1k reference tokens beyond 4,096 per shot. Not modelled.

## Price Variables (microcents)
- `per_request`: 50000000
- `rate_480p_per_sec`: 5000000, `rate_768p_per_sec`: 8000000, `rate_1080p_per_sec`: 16000000
- `per_generated_reference_image`: 10000000

## Expression
`outputs[0].seconds` = ceil(output seconds) with a 5s floor (shots are not reported, so one shot is assumed).
`outputs[0].extra.generated_reference_images` = task_inputs.max_generated_reference_images when no reference_images were given, else 0.
cost = per_request + seconds * rate(resolution) + generated_reference_images * per_generated_reference_image.
Estimate is a range (duration unknown, up to 15s).

## Expected values
- 5s 768P, own references: $0.90 = 90,000,000
- 5s 768P, no references, 2 generated: $1.10 = 110,000,000
