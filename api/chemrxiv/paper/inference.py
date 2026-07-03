import logging
import requests
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field


class AppInput(BaseAppInput):
    doi: str = Field(description="chemrxiv paper DOI like 10.26434/chemrxiv-2023-fw8n4")


class AppOutput(BaseAppOutput):
    paper: dict = Field(description="paper metadata with title, authors, abstract, dates, references")


class App(BaseApp):

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("chemrxiv paper app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"fetching chemrxiv paper: {input_data.doi}")

        response = requests.get(
            f"https://api.crossref.org/works/{input_data.doi}",
            headers={"User-Agent": "inference.sh (mailto:ok@inference.sh)"},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        item = data.get("message", {})

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
            "reference_count": item.get("reference-count", 0),
            "is_referenced_by_count": item.get("is-referenced-by-count", 0),
            "license": [l.get("URL", "") for l in item.get("license", [])],
        }

        self.logger.info(f"fetched: {paper['title'][:80]}")
        return AppOutput(paper=paper)
