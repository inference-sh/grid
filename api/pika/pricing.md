# Pika API pricing (dev.pika.art)

Source: https://dev.pika.art/pricing and `GET https://api.dev.pika.art/catalog/apis` (`display_pricing`), read 2026-11-08.
Pika quotes in micro-USD ($1 = 1,000,000 micro-USD). Platform pricing uses microcents ($1 = 100,000,000 microcents), so
`microcents = micro_usd * 100`. Every completed job reports `billing.charge_micro_usd`; the apps copy it into
`outputs[0].extra.charge_micro_usd` so a charge can be checked against the CEL result.

**Launch promo until 2025-11-15:** the catalog currently shows half of the regular rates below. Price the apps on the
regular rates; the first invoice after the promo ends is the check.

## pika/pika-h3-flex (text-to-video, image-to-video, first/last frame)

Per output second. `outputs[0].resolution` is the requested value, lower case: `480p`, `768p`, `1080p`, `2k`, `4k`.
`outputs[0].seconds` is `usage.output_seconds` when Pika reports it, else the probed file length.

| Resolution | Regular $/s | Microcents/s | Launch $/s |
|---|---|---|---|
| 480p | 0.02 | 2,000,000 | 0.01 |
| 768p | 0.04 | 4,000,000 | 0.02 |
| 1080p | 0.07 | 7,000,000 | 0.035 |
| 2k | 0.08 | 8,000,000 | 0.04 |
| 4k | 0.16 | 16,000,000 | 0.08 |

Audio stems (`audio_stems: true`): `outputs[1..3]` are `AudioMeta` with `extra.stem` in dialogue/music/effect; billed
$0.001/s (100,000 microcents) on the sum of their `seconds`, no channel multiplier. Estimate: 3 x duration x $0.001.

## pika/pika-h3-flex-reference (reference-to-video)

Per output second **plus per input reference-video second**, both at the reference rate. Image and audio references are
free. `inputs[]` carries a `VideoMeta` with probed `seconds` per reference video; `outputs[0].extra.input_video_seconds`
is their sum.

| Resolution | Regular $/s | Microcents/s | Launch $/s |
|---|---|---|---|
| 480p | 0.06 | 6,000,000 | 0.03 |
| 768p | 0.12 | 12,000,000 | 0.06 |
| 1080p | 0.15 | 15,000,000 | 0.075 |
| 2k | 0.16 | 16,000,000 | 0.08 |
| 4k | 0.20 | 20,000,000 | 0.10 |

Stems as above.

## pika/pika-h3-flex-upscale (draft-to-video with an external video)

Per output second at `rate[target] - rate[source tier]` from the generation table. The source tier is chosen from the
probed short edge: <= 480 px is `480p`, otherwise `768p`. `inputs[0].resolution` holds the tier,
`outputs[0].extra.source_tier` repeats it.

| Source -> target | Regular $/s | Microcents/s |
|---|---|---|
| 480p -> 1080p | 0.05 | 5,000,000 |
| 480p -> 2k | 0.06 | 6,000,000 |
| 480p -> 4k | 0.14 | 14,000,000 |
| 768p -> 1080p | 0.03 | 3,000,000 |
| 768p -> 2k | 0.04 | 4,000,000 |
| 768p -> 4k | 0.12 | 12,000,000 |

## pika/pika-h3-flex-audio-stems

$0.001 per output stem second (100,000 microcents). `outputs[]` is one `AudioMeta` per stem; bill the sum of `seconds`.
Estimate from `inputs[0].seconds` x 3 (Dialogue, Music, Effect) x $0.001; exact only after the run.

## pika/pika-h3-flex-oss (self-hosted open weights, native/generative/video/pika-h3-flex-oss)

Priced by GPU time (`inference_expression: resource_fee`, "GPU time, billed per millisecond"), like the other
80 GB GPU apps. No per-second rate table; the cost is the engine's GPU wall clock including setup on a cold
engine. Measured on the first cloud runs (480p, 5.17 s, group-offloaded transformer, weights cached on the engine):

| Run | Setup | Generation | Peak GPU |
|---|---|---|---|
| text-to-video 480p 5 s | 52 s warm (173 s with the 150 GB download) | 158 s | 62.2 GiB |
| image-to-video 480p 5 s | 52 s | 165 s | 72.2 GiB |
| image-to-video 768p 15 s (1344x768, 362 frames, the heaviest setting) | 51 s | 1100 s | 72.2 GiB |

Fresh task after submit and publish: 2c4kn9pt7j3kydemcv9fwjmscn charged 0, which is expected for the infsh team.
`usage/cel.go`: a team's own engine prices its resources at 0, so runs on the platform's own GPU fleet stay
free for infsh (infsh/harrier-oss-v1 shows the same). Other workspaces pay the GPU reservation rate per
millisecond; `belt app estimate` reports `confidence: unknown, depends_on: [resource_fee]`.

## Confirmed on 2026-11-08

- Pika bills measured output seconds: a 3 s request produced 73 frames = 3.0417 s (`usage.output_seconds`).
- Generation runs charged 0 micro-USD upstream during the launch promo (15 free API generations a day).
- Upscale 480p -> 1080p, 3.0417 s: Pika charged 76,042 micro-USD = $0.025/s, the launch rate for 0.035 - 0.010.
- Stems, 3 x 3.072 s: Pika charged 9,216 micro-USD = exactly $0.001 per stem second.
- Published pricing (regular rates): pika-h3-flex 480p 3.04 s task billed 6,083,333 microcents;
  audio-stems 9.216 s billed 921,600 microcents.

## Not wrapped

`pika/pika-h3-flex/prompt-rewriter` ($0.02 per request) is covered by the `rewrite` flag on the generation apps.
