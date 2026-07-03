import logging
import requests
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    doi: str = Field(description="paper DOI like 10.1101/2023.01.01.123456")
    server: Optional[str] = Field(default="biorxiv", description="server: biorxiv or medrxiv")


class AppOutput(BaseAppOutput):
    paper: dict = Field(description="paper metadata")
    raw: dict = Field(description="raw api response")


class App(BaseApp):

    async def setup(self):
        self.base_url = "https://api.biorxiv.org/details"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("biorxiv paper app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        server = input_data.server or "biorxiv"
        url = f"{self.base_url}/{server}/{input_data.doi}/na/json"

        self.logger.info(f"fetching {server} paper: {input_data.doi}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        collection = data.get("collection", [])

        if not collection:
            raise ValueError(f"paper not found: {input_data.doi}")

        paper = collection[0]
        self.logger.info(f"fetched paper: {paper.get('title', '')[:80]}")

        return AppOutput(paper=paper, raw=data)
