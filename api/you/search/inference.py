"""
You.com Web Search API.

Ground your apps in reliable, web-scale knowledge with contextual snippets.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, RawMeta
from pydantic import Field, BaseModel
from typing import Optional, List

from .you_helper import get_api_key, setup_logger, search_request


class SearchHit(BaseModel):
    title: str = Field(description="Page title")
    url: str = Field(description="Page URL")
    snippet: str = Field(description="Contextual snippet")


class AppInput(BaseAppInput):
    query: str = Field(description="Search query")
    count: int = Field(default=10, ge=1, le=20, description="Number of results (1-20)")
    country: Optional[str] = Field(default=None, description="ISO 3166-1 alpha-2 country code for location targeting")
    search_lang: Optional[str] = Field(default=None, description="BCP 47 language code to filter results")
    livecrawl: bool = Field(default=False, description="Enable live crawling for full page content (adds latency, billed separately)")
    include_domains: Optional[List[str]] = Field(default=None, description="Only return results from these domains")
    exclude_domains: Optional[List[str]] = Field(default=None, description="Exclude results from these domains")
    boost_domains: Optional[List[str]] = Field(default=None, description="Boost results from these domains")
    recent_past_day: bool = Field(default=False, description="Only results from the past day")
    recent_past_week: bool = Field(default=False, description="Only results from the past week")
    recent_past_month: bool = Field(default=False, description="Only results from the past month")


class AppOutput(BaseAppOutput):
    results: List[SearchHit] = Field(description="Search results")
    raw: dict = Field(description="Full API response")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("You.com Search initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = search_request(
                api_key=self.api_key,
                query=input_data.query,
                count=input_data.count,
                country=input_data.country,
                search_lang=input_data.search_lang,
                livecrawl=input_data.livecrawl,
                include_domains=input_data.include_domains,
                exclude_domains=input_data.exclude_domains,
                boost_domains=input_data.boost_domains,
                recent_past_day=input_data.recent_past_day,
                recent_past_week=input_data.recent_past_week,
                recent_past_month=input_data.recent_past_month,
                logger=self.logger,
            )

            hits = []
            for hit in response.get("hits", []):
                snippets = hit.get("snippets", [])
                hits.append(SearchHit(
                    title=hit.get("title", ""),
                    url=hit.get("url", ""),
                    snippet="\n".join(snippets) if snippets else hit.get("description", ""),
                ))

            self.logger.info(f"Returned {len(hits)} results")

            # $5 per 1000 calls = $0.005 per call = 0.5 cents
            cost_cents = 0.5
            if input_data.livecrawl:
                # +$1 per 1000 pages = $0.001 per page
                cost_cents += input_data.count * 0.001 * 100  # convert to cents

            return AppOutput(
                results=hits,
                raw=response,
                output_meta=OutputMeta(outputs=[RawMeta(cost=cost_cents)]),
            )

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {str(e)}")
