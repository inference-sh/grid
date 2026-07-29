# Mirage Pricing

Mirage publishes usage-based pricing for the two `api.mirage.app` features. The three legacy
`api.captions.ai` features have **no published API rates** — see below.

## Video Generation (video-1)

| Model | Basis | Price |
|-------|-------|-------|
| Mirage Video 1 | Length of the **generated output** video | **$0.175 per second** |

Rounded **up to the nearest 6-second increment**. A 10.2-second result bills as 12 seconds.

Output duration tracks the input audio length, so the audio clip you supply determines the cost.

## Video Captions (video-captions)

| Feature | Basis | Price |
|---------|-------|-------|
| Add Captions | Length of the **input** video | **$0.15 per minute** |

Rounded **up to the nearest minute**. Captioning a 2.5-minute video bills as 3 minutes ($0.45).

Note the asymmetry with generation: captions meter the **input**, generation meters the **output**.

## AI Creator, AI Ads, AI Twin

Not published. Mirage's pricing page documents only the two features above, and the legacy
endpoints predate it. The public $9.99–$69.99/mo Captions plans and their credit allowances are
**consumer app pricing and do not describe API billing** — do not derive rates from them.

To price these three, either get rates from the Mirage account manager, or measure: run one job,
read the account's usage delta, and set the CEL expression from the measured cost.

Until a rate is established, these three apps report `output_meta` with output video duration
(and input media counts for `ai-ads` / `ai-twin`) so that a duration- or count-based expression
can be dropped in without redeploying.

## Rate Limits

| Feature | Limit |
|---------|-------|
| Video Generation | **2 requests/min** |
| Video Captions | **100 requests/min** |

Per organization. Exceeding either returns HTTP 429 with
`{"error": {"type": "rate_limit_exceeded"}}`, surfaced by `mirage_helper.py` as a message naming
the 2/min generation cap.

## Billing Rules

- **Failed jobs:** a job ending `FAILED` or `CANCELLED` raises before any output is produced.
- **Metering source:** `video-1` and `video-captions` probe the actual output file with ffprobe
  rather than trusting a requested resolution, so a job that silently returns something shorter
  than asked for is not over-billed. `video-1` falls back to the input audio duration only if
  ffprobe cannot read the output at all.

## Examples

A 30-second talking-head video from a 30-second audio clip:
- 30s → already a 6s multiple → $0.175 × 30 = **$5.25**

A 10.2-second video:
- rounds to 12s → $0.175 × 12 = **$2.10**

Captioning a 90-second vertical video:
- rounds to 2 minutes → $0.15 × 2 = **$0.30**
