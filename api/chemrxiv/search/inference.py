import logging
import requests
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional


CHEMRXIV_PREFIX = "10.26434"


class AppInput(BaseAppInput):
    query: str = Field(description="search keywords")
    rows: int = Field(default=10, ge=1, le=100, description="number of results")
    sort: Optional[str] = Field(default="relevance", description="sort: relevance, published-desc, published-asc")


class AppOutput(BaseAppOutput):
    results: List[dict] = Field(description="list of papers with title, authors, doi, abstract, published date")
    total: int = Field(description="total matching results")


class App(BaseApp):

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("chemrxiv search app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"searching chemrxiv: {input_data.query}")

        sort_map = {
            "relevance": "relevance",
            "published-desc": "published",
            "published-asc": "published",
        }
        order_map = {
            "relevance": "desc",
            "published-desc": "desc",
            "published-asc": "asc",
        }

        params = {
            "query": input_data.query,
            "rows": input_data.rows,
            "sort": sort_map.get(input_data.sort, "relevance"),
            "order": order_map.get(input_data.sort, "desc"),
        }

        response = requests.get(
            f"https://api.crossref.org/prefixes/{CHEMRXIV_PREFIX}/works",
            params=params,
            headers={"User-Agent": "inference.sh (mailto:ok@inference.sh)"},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        items = message.get("items", [])
        total = message.get("total-results", 0)

        papers = []
        for item in items:
            paper = {
                "title": item.get("title", [""])[0] if item.get("title") else "",
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item.get("author", [])
                ],
                "doi": item.get("DOI", ""),
                "url": item.get("URL", ""),
                "abstract": item.get("abstract", ""),
                "published": item.get("created", {}).get("date-time", ""),
                "type": item.get("type", ""),
                "subject": item.get("subject", []),
            }
            papers.append(paper)

        self.logger.info(f"found {len(papers)} papers out of {total} total")
        return AppOutput(results=papers, total=total)
