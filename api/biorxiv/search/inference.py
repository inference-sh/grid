import logging
import requests
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional


class AppInput(BaseAppInput):
    start_date: str = Field(description="start date in YYYY-MM-DD format")
    end_date: str = Field(description="end date in YYYY-MM-DD format")
    cursor: int = Field(default=0, ge=0, description="pagination offset")
    category: Optional[str] = Field(default=None, description="filter by subject like cell_biology, neuroscience")
    server: Optional[str] = Field(default="biorxiv", description="server to search: biorxiv or medrxiv")


class AppOutput(BaseAppOutput):
    results: List[dict] = Field(description="list of preprint papers")
    total: int = Field(description="total number of results")
    cursor: int = Field(description="current cursor position")


class App(BaseApp):

    async def setup(self):
        self.base_url = "https://api.biorxiv.org/details"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("biorxiv search app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        server = input_data.server or "biorxiv"
        url = f"{self.base_url}/{server}/{input_data.start_date}/{input_data.end_date}/{input_data.cursor}/json"

        params = {}
        if input_data.category:
            params["category"] = input_data.category

        self.logger.info(f"searching {server}: {input_data.start_date} to {input_data.end_date}, cursor={input_data.cursor}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        messages = data.get("messages", [])
        total = 0
        if messages:
            total = int(messages[0].get("total", 0))

        results = data.get("collection", [])
        self.logger.info(f"found {len(results)} preprints out of {total} total")

        return AppOutput(
            results=results,
            total=total,
            cursor=input_data.cursor,
        )
