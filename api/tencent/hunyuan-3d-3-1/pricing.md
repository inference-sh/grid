# Pricing

What Tencent Cloud charges for Hunyuan 3D Pro (Model 3.1), so the app's price can be set from it.

- source_url: https://www.tencentcloud.com/document/product/1284/75281
- checked: 2026-10-28
- confidence: medium. Postpaid $0.02 per credit; prepaid packs $0.015 to $0.0135. The international API does not report credits, so the app computes them from the request with the table below.

`run` reports `outputs[0].extra.credits` (computed), plus `generate_type`, `enable_pbr`, `multiview` and `face_count_set`. Price it as credits x rate; the estimate is exact from the inputs.

| item | credits |
|------|--------:|
| GenerateType Normal | 25 |
| GenerateType Geometry | 15 |
| MultiViewImages | +10 |
| EnablePBR | +10 |
| FaceCount set | +10 |

Normal with PBR: 35 credits, $0.70 postpaid.
