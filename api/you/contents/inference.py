"""
You.com Contents API.

Fetch clean markdown or HTML from any URL.
Batch up to 10 pages per request.
"""

from enum import Enum
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, RawMeta
from pydantic import Field, BaseModel
from typing import Optional, List

from .you_helper import get_api_key, setup_logger, contents_request


class FormatEnum(str, Enum):
    markdown = "markdown"
    html = "html"


class PageContent(BaseModel):
    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Page title")
    content: str = Field(default="", description="Extracted content in requested format")


class AppInput(BaseAppInput):
    urls: List[str] = Field(description="URLs to fetch (up to 10)", min_length=1, max_length=10)
    format: FormatEnum = Field(default=FormatEnum.markdown, description="Output format: markdown or html")
    crawl_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Crawl timeout in seconds (1-60). Use 20-30 for JS-heavy pages."
    )


class AppOutput(BaseAppOutput):
    pages: List[PageContent] = Field(description="Extracted page contents")
    raw: list = Field(description="Full API response")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("You.com Contents initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = contents_request(
                api_key=self.api_key,
                urls=input_data.urls,
                format=input_data.format.value,
                crawl_timeout=input_data.crawl_timeout,
                logger=self.logger,
            )

            pages = []
            items = response if isinstance(response, list) else response.get("items", response.get("contents", []))
            for item in items:
                content = item.get(input_data.format.value, item.get("content", ""))
                pages.append(PageContent(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=content,
                ))

            self.logger.info(f"Fetched {len(pages)} page(s)")

            # $1 per 1000 pages = $0.001 per page = 0.1 cents per page
            cost_cents = len(input_data.urls) * 0.1

            return AppOutput(
                pages=pages,
                raw=response,
                output_meta=OutputMeta(outputs=[RawMeta(cost=cost_cents)]),
            )

        except Exception as e:
            self.logger.error(f"Contents fetch failed: {e}")
            raise RuntimeError(f"Contents fetch failed: {str(e)}")
