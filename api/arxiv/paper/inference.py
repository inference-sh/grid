import logging
import requests
import xml.etree.ElementTree as ET
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List


class AppInput(BaseAppInput):
    arxiv_id: str = Field(description="arxiv paper id like 2301.07041 or 2301.07041v1")


class AppOutput(BaseAppOutput):
    paper: dict = Field(description="paper metadata with title, authors, summary, published, updated, arxiv_id, pdf_url, categories, links")


class App(BaseApp):

    async def setup(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("arxiv paper app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"fetching arxiv paper: {input_data.arxiv_id}")

        params = {"id_list": input_data.arxiv_id}
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entry = root.find("atom:entry", ns)
        if entry is None:
            raise ValueError(f"paper not found: {input_data.arxiv_id}")

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

        links = []
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            link_data = {
                "href": link.get("href", ""),
                "rel": link.get("rel", ""),
                "type": link.get("type", ""),
                "title": link.get("title", ""),
            }
            links.append(link_data)
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")

        paper = {
            "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
            "authors": authors,
            "summary": (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip(),
            "published": entry.findtext("atom:published", default="", namespaces=ns),
            "updated": entry.findtext("atom:updated", default="", namespaces=ns),
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "categories": categories,
            "links": links,
        }

        self.logger.info(f"fetched paper: {paper['title'][:80]}")
        return AppOutput(paper=paper)
