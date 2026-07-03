import logging
import requests
import xml.etree.ElementTree as ET
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional


class AppInput(BaseAppInput):
    query: str = Field(description="search query. supports field prefixes: ti: (title), au: (author), abs: (abstract), cat: (category), all: (all fields). boolean operators: AND, OR, ANDNOT")
    max_results: int = Field(default=10, ge=1, le=100, description="number of results to return")
    start: int = Field(default=0, ge=0, description="pagination offset")
    sort_by: Optional[str] = Field(default="relevance", description="sort by: relevance, lastUpdatedDate, or submittedDate")
    sort_order: Optional[str] = Field(default="descending", description="sort order: ascending or descending")
    category: Optional[str] = Field(default=None, description="filter by arxiv category like cs.AI, physics.hep-th")


class AppOutput(BaseAppOutput):
    results: List[dict] = Field(description="list of papers with title, authors, summary, published, updated, arxiv_id, pdf_url, categories")
    total: int = Field(description="total number of results")


class App(BaseApp):

    async def setup(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("arxiv search app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        search_query = input_data.query
        if input_data.category and f"cat:" not in search_query:
            search_query = f"cat:{input_data.category} AND {search_query}"

        params = {
            "search_query": search_query,
            "start": input_data.start,
            "max_results": input_data.max_results,
            "sortBy": input_data.sort_by,
            "sortOrder": input_data.sort_order,
        }

        self.logger.info(f"searching arxiv: query={search_query}, max_results={input_data.max_results}")
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        total_results = int(root.findtext("opensearch:totalResults", default="0", namespaces=ns))

        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.findtext("atom:id", default="", namespaces=ns)
            if arxiv_id:
                arxiv_id = arxiv_id.split("/abs/")[-1]

            authors = [
                author.findtext("atom:name", default="", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]

            categories = [
                cat.get("term", "")
                for cat in entry.findall("atom:category", ns)
            ]

            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            paper = {
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                "authors": authors,
                "summary": (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip(),
                "published": entry.findtext("atom:published", default="", namespaces=ns),
                "updated": entry.findtext("atom:updated", default="", namespaces=ns),
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url,
                "categories": categories,
            }
            papers.append(paper)

        self.logger.info(f"found {len(papers)} papers out of {total_results} total")
        return AppOutput(results=papers, total=total_results)
