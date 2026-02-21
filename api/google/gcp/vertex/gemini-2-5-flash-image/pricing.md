# Gemini 2.5 Flash Image — Pricing Reference

## Token Rates (Standard, Vertex AI)

| Type | Price (/1M tokens) |
|------|--------------------|
| Input (text, image, video) | $0.30 |
| Audio input | $1.00 |
| Text output (response + reasoning) | $2.50 |
| Image output | $30.00 |

**Image output tokens**: 1024×1024 = 1,290 tokens. Count varies by resolution.

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
prompt_token_count: 500     → Input:       500 × $0.30/1M  = $0.00015
thoughts_token_count: 200   → Text output: 200 × $2.50/1M  = $0.0005
Image: 1024×1024            → Image output: 1290 × $30/1M  = $0.0387
                                                     Total ≈ $0.039
```

## Example: Blocked by safety filter

```
prompt_token_count: 500     → Input:       $0.00015 (still billed)
thoughts_token_count: 200   → Text output: $0.0005  (still billed)
candidates_token_count: 0   → No image charge
                                      Total ≈ $0.001
```

## Notes

- Reasoning tokens are **always billed**, even on failed generations
- Google Search grounding: 1,500 grounded prompts/day free (shared with 2.0 Flash), then $35/1,000 prompts
