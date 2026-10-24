# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: <the vendor's pricing page>
- checked: <YYYY-MM-DD>
- confidence: <high | medium | low: how sure the rates below are, and why>

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| scrape | | request | POST /web/scrape |
| crawl | | request | POST /web/crawl |
| search | | request | POST /web/search |
| map_urls | | request | GET /web/urls |
| extract | | request | POST /web/extract |
| answer | | request | POST /web/answers |
| styleguide | | request | GET /web/styleguide |
| fonts | | request | GET /web/fonts |
| naics | | request | GET /web/naics |
| sic | | request | GET /web/sic |
| brand | | request | POST /brand/retrieve |
| brand_search | | request | GET /brand/search |
| company_search | | request | POST /company/search |
| people_search | | request | POST /people/search |
| people_enrich | | request | POST /people/enrich |
| news_search | | request | POST /news/search |

A metered function reports what it consumed in `output_meta` (for example
`OutputMeta(outputs=[RawMeta(cost=...)])`), and the app's price is set from that.
A free function reports `OutputMeta(inputs=[], outputs=[])`.
