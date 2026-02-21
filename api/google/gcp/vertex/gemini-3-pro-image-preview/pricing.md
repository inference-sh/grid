# Gemini 3 Pro Image Preview — Pricing Reference

## Token Rates (Standard, Vertex AI)

| Type | ≤200K input | >200K input |
|------|-------------|-------------|
| Input (text, image, video, audio) | $2.00/1M | $4.00/1M |
| Text output (response + reasoning) | $12.00/1M | $18.00/1M |
| Image output | $120.00/1M | N/A |

**Image output tokens**: 1K/2K image = 1,120 tokens (**$0.134/image**). 4K image = 2,000 tokens (**$0.24/image**).

## OutputMeta → Billing

```
OutputMeta(
    inputs=[TextMeta(tokens=prompt_token_count)],         # "Input" rate
    outputs=[
        TextMeta(tokens=thoughts_token_count),             # "Text output" rate (reasoning)
        TextMeta(tokens=candidates_token_count),           # "Text output" rate (response)
        ImageMeta(width=W, height=H),                      # "Image output" rate
    ]
)
```

## Example: Successful 1K image

```
prompt_token_count: 718     → Input:       718 × $2/1M   = $0.0014
thoughts_token_count: 292   → Text output: 292 × $12/1M  = $0.0035
Image: 1024×1024            → Image output: 1120 × $120/1M = $0.1344
                                                    Total ≈ $0.139
```

## Example: Blocked by safety filter

```
prompt_token_count: 718     → Input:       $0.0014 (still billed)
thoughts_token_count: 292   → Text output: $0.0035 (still billed)
candidates_token_count: 0   → No image charge
                                     Total ≈ $0.005
```

## Notes

- Reasoning tokens are **always billed**, even on failed generations
- Google Search grounding: 5,000 queries/month free, then $14/1,000 queries
