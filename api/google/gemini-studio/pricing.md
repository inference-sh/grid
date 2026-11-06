# Gemini Studio pricing

Source: https://ai.google.dev/gemini-api/docs/pricing (read 2026-11-06). $1 = 100,000,000 microcents.

## Gemini Omni 1.2 Flash (`google/gemini-omni-1-2-flash`)

Google bills tokens. Paid tier only.

| Item | Rate per 1M tokens | Microcents per token |
| :--- | :--- | :--- |
| Input (text, image) | $1.50 | 150 |
| Output, text and thinking | $9.00 | 900 |
| Output, video | $12.00 | 1,200 |

Video output tokens are fixed per second of requested length (confirmed on live runs at every resolution):

| Resolution | Tokens per second | $ per second | Microcents per second |
| :--- | :--- | :--- | :--- |
| 360p | 1,931 | $0.023172 | 2,317,200 |
| 720p | 5,792 | $0.069504 | 6,950,400 |
| 1080p | 8,688 | $0.104256 | 10,425,600 |
| 4k | 17,376 | $0.208512 | 20,851,200 |

Measured on live runs: an input image is about 1,080 to 1,100 tokens; thinking is 300 to 1,250 tokens per run.

### output_meta

- `inputs[0]` text: `tokens` = `total_input_tokens` (prompt and images).
- `outputs[0]` video: `resolution` (`360p`, `720p`, `1080p`, `4k`), `seconds` = requested duration, `extra.tokens` = video output tokens, `extra.mode`.
- `outputs[1]` text: `tokens` = text output plus thinking tokens.

### Expected charges at upstream rates

| Run | Input tok | Video tok | Thinking tok | Microcents | $ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 4k, 3s, text only | 17 | 52,128 | 324 | 62,847,750 | 0.6285 |
| 360p, 5s, image + 1 reference | 2,248 | 9,655 | 1,157 | 12,964,500 | 0.1296 |
| 360p, 15s, text only | 19 | 28,965 | 1,216 | 35,855,250 | 0.3586 |

Estimate: exact for the video part (`duration` x rate for `resolution`); input and thinking tokens add a small range on top.

### Published (2026-11-06)

Pass-through at the upstream rates, no markup. Store version `16kmhsbve15p78gbzh9zxbk7tf`, visibility public.

| Price variable | Microcents |
| :--- | :--- |
| `input_per_million` | 150,000,000 |
| `text_output_per_million` | 900,000,000 |
| `per_second_360p` | 2,317,200 |
| `per_second_720p` | 6,950,400 |
| `per_second_1080p` | 10,425,600 |
| `per_second_4k` | 20,851,200 |

Estimates (confidence `range`): default input (720p, 10s) 69,504,000 to 73,254,000; 360p 3s 6,951,600 to 10,701,600; 4k 15s with 3 images 312,768,000 to 317,058,000.

Billed check: task `5apv3kvpczwxg5cfws5skymyb9` (360p, 3s, 20 input tokens, 629 thinking tokens) charged 7,520,700 microcents, equal to 3 x 2,317,200 + 20 x 150 + 629 x 900.
