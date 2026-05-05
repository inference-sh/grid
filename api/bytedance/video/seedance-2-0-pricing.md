# Seedance 2.0 Series Pricing (BytePlus ModelArk)

## Resource Packs (Prepaid)

Resource packs are prepaid tokens used to deduct consumption from online inference. Purchase at [BytePlus Console](https://console.byteplus.com/common-buy/ModelArk%7C%7Cd7d6aanpgiftptb9ajcg).

> **Non-refundable.** If all packs expire or are depleted, excess consumption auto-converts to pay-as-you-go.

### Seedance 2.0

| Specification | Price (USD) | Min Purchase | Expires |
|---|---|---|---|
| 1M tokens | $4.30 | 7 packs minimum | 90 days |
| 10M tokens | $43.00 | No limit | 90 days |
| 100M tokens | $430.00 | No limit | 90 days |

### Seedance 2.0 Fast

| Specification | Price (USD) | Min Purchase | Expires |
|---|---|---|---|
| 1M tokens | $3.30 | 9 packs minimum | 90 days |
| 10M tokens | $33.00 | No limit | 90 days |
| 100M tokens | $330.00 | No limit | 90 days |

## Pay-As-You-Go Token Prices

### Seedance 2.0

| Input Type | Resolution | USD/K tokens | Pack Deduction Ratio |
|---|---|---|---|
| Video input included | 480p / 720p | $0.0043 | 1 : 1 |
| Video input not included | 480p / 720p | $0.0070 | 1 : 1.6279 |
| Video input included | 1080p | $0.0047 | 1 : 1.0930 |
| Video input not included | 1080p | $0.0077 | 1 : 1.7907 |

### Seedance 2.0 Fast

| Input Type | Resolution | USD/K tokens | Pack Deduction Ratio |
|---|---|---|---|
| Video input included | 480p / 720p | $0.0033 | 1 : 1 |
| Video input not included | 480p / 720p | $0.0056 | 1 : 1.6970 |

> Seedance 2.0 Fast does not support 1080p.

## Token Deduction Rules

- When **video input is included**, 1 consumed token = 1 resource pack token deducted (base rate).
- When **no video input**, the higher unit price means a multiplier applies (see deduction ratio above).
- Token formula: `Token Consumption ≈ (Width × Height × Frame Rate × Duration) / 1024`

## Deduction Priority

1. Resource packs deducted first (earliest-expiring pack first, then earliest-purchased).
2. If all packs are expired or depleted, excess auto-converts to pay-as-you-go.

## Key Notes

- Packs activate immediately on purchase (no manual activation needed).
- 90-day validity from purchase date; remaining tokens invalidate on expiry.
- Concurrent tasks: 10 per model.
- RPM: 600 requests/minute.
