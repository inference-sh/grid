# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: <the vendor's pricing page>
- checked: <YYYY-MM-DD>
- confidence: <high | medium | low: how sure the rates below are, and why>

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| answer | | request | POST /answer |
| get_contents | | request | POST /contents |
| find_similar | | request | POST /findSimilar |
| search | | request | POST /search |

A metered function reports what it consumed in `output_meta` (for example
`OutputMeta(outputs=[RawMeta(cost=...)])`), and the app's price is set from that.
A free function reports `OutputMeta(inputs=[], outputs=[])`.
