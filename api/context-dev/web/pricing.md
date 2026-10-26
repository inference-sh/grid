# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: https://www.context.dev/pricing and https://docs.context.dev/account/credits
- checked: 2026-10-25
- confidence: high: the credit table is published per operation; each call logs the credits context.dev reports (key_metadata.credits_consumed) so the price can be checked against it

Credit price by plan: Developer $20/month for 10,000 credits ($0.002 each, overage $3.00 per 1,000); Pro $90/month for 100,000 ($0.0009 each, overage $2.00 per 1,000). Validation, auth, 408, 429 and server errors are not charged.

Price from the input and from what the call returned, never from the vendor's reported charge. A function whose cost depends on its result reports the count as a raw output item, `output_meta.outputs[0].extra`. The same item carries `upstream_credits`, the credits context.dev reports for the call: informational only, never used in a price.

| function | upstream rate | priced from | notes |
|----------|---------------|-------------|-------|
| scrape | 1 credit; 2 with sharedParams.actions; +4 with formats.json, +1 formats.highlights, +1 formats.product +1 per PDF page OCR recovers (sharedParams.parsers.pdf.ocr=auto): from `ocr_pages` when the response carries it, else every page in startPage..endPage, else unpriced | input, output | |
| crawl | 1 credit per page returned | outputs[0].extra.pages | |
| search | 1 credit per 10 results returned, rounded up; +1 per 10 each with markdownOptions or highlightsOptions | outputs[0].extra.results, input | |
| map_urls | 1 credit; 2 with search | input | |
| extract | 10 credits | flat | |
| answer | 10 credits with mode fast; 100 otherwise (ultra is the default) | input | |
