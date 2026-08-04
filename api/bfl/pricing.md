# BFL Official Pricing (FLUX 3 Video)

Per-second pricing. Partial seconds rounded up to next whole second.

## Resolution Bands
- **HD**: Above 0.5 and up to 1.0 MP per frame (e.g. 1280x704, 960x960, 1440x608)
- **FHD (Full HD)**: Above 1.0 and up to 2.0 MP per frame (e.g. 1920x1088, 1440x1440, 2176x928)

## Video Generation

### Text/Image-to-Video (t2v, i2v)

| Model | HD | FHD |
|-------|-----|------|
| **FLUX 3 Video Draft** | $0.06/s | — |
| **FLUX 3 Video** | $0.17/s | $0.29/s |

### Video-to-Video (v2v)

| Model | HD | FHD |
|-------|-----|------|
| **FLUX 3 Video Draft** | $0.12/s | — |
| **FLUX 3 Video** | $0.41/s | $0.53/s |

Audio included at no extra charge. Clips up to 20 seconds.

## OutputMeta Mapping

- `outputs[0].seconds` — video duration in seconds
- `outputs[0].width`, `outputs[0].height` — resolution
- `outputs[0].extra.model` — "flux-3-video"
- `outputs[0].extra.mode` — "t2v", "i2v", "v2v", or "draft_enhance"
- `outputs[0].extra.resolution` — "hd" or "fhd"
- `outputs[0].extra.draft` — true/false

## Microcent Conversion

$1.00 = 100,000,000 microcents

| Per-unit price | Microcents/second |
|---|---|
| $0.06/s (draft, HD, t2v/i2v) | 6,000,000 |
| $0.12/s (draft, HD, v2v) | 12,000,000 |
| $0.17/s (HD, t2v/i2v) | 17,000,000 |
| $0.29/s (FHD, t2v/i2v) | 29,000,000 |
| $0.41/s (HD, v2v) | 41,000,000 |
| $0.53/s (FHD, v2v) | 53,000,000 |

## CEL Expression Template

### Price variables
```
draft_hd_t2v_per_second = 6,000,000
draft_hd_v2v_per_second = 12,000,000
hd_t2v_per_second = 17,000,000
fhd_t2v_per_second = 29,000,000
hd_v2v_per_second = 41,000,000
fhd_v2v_per_second = 53,000,000
```

### partner_expression
```cel
get_bool(outputs[0], "extra.draft", false)
  ? (get_string(outputs[0], "extra.mode", "t2v") == "v2v"
      ? double(outputs[0].seconds) * double(prices.draft_hd_v2v_per_second)
      : double(outputs[0].seconds) * double(prices.draft_hd_t2v_per_second))
  : (get_string(outputs[0], "extra.mode", "t2v") == "v2v"
      ? (get_string(outputs[0], "extra.resolution", "hd") == "fhd"
          ? double(outputs[0].seconds) * double(prices.fhd_v2v_per_second)
          : double(outputs[0].seconds) * double(prices.hd_v2v_per_second))
      : (get_string(outputs[0], "extra.resolution", "hd") == "fhd"
          ? double(outputs[0].seconds) * double(prices.fhd_t2v_per_second)
          : double(outputs[0].seconds) * double(prices.hd_t2v_per_second)))
```

### total_expression
```cel
partner_fee
```

### description
```cel
"From $" + string(to_cents(prices.hd_t2v_per_second)) + "/sec (HD)"
```
