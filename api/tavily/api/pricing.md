# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: <the vendor's pricing page>
- checked: <YYYY-MM-DD>
- confidence: <high | medium | low: how sure the rates below are, and why>

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| crawl | | request | POST /crawl |
| extract | | request | POST /extract |
| post_feedback | | request | POST /feedback |
| post_logs | | request | POST /logs |
| map | | request | POST /map |
| create_research | | request | POST /research |
| get_research | | request | GET /research/{request_id} |
| search | | request | POST /search |
| get_usage | | request | GET /usage |

A metered function reports what it consumed in `output_meta` (for example
`OutputMeta(outputs=[RawMeta(cost=...)])`), and the app's price is set from that.
A free function reports `OutputMeta(inputs=[], outputs=[])`.
