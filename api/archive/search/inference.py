from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
import requests
import logging

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    query: str = Field(
        description="search query using lucene-like syntax"
    )
    rows: int = Field(
        default=10,
        ge=1,
        le=100,
        description="results per page"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="page number"
    )
    fields: Optional[List[str]] = Field(
        default=None,
        description="metadata fields to return (default: identifier, title, description, mediatype, date, creator)"
    )
    sort: Optional[str] = Field(
        default=None,
        description="sort field like 'downloads desc' or 'date asc'"
    )
    mediatype: Optional[str] = Field(
        default=None,
        description="filter by type: texts, movies, audio, software, image, etc."
    )


class AppOutput(BaseAppOutput):
    results: list = Field(description="search results")
    total: int = Field(description="total matching items")
    raw: dict = Field(description="full api response")


class App(BaseApp):
    async def setup(self, metadata):
        logging.info("archive search app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        fields = input_data.fields or [
            "identifier", "title", "description", "mediatype", "date", "creator"
        ]

        params = [
            ("q", input_data.query),
            ("output", "json"),
            ("rows", input_data.rows),
            ("page", input_data.page),
        ]

        for field in fields:
            params.append(("fl[]", field))

        if input_data.sort:
            params.append(("sort[]", input_data.sort))

        if input_data.mediatype:
            params[0] = ("q", f"{input_data.query} AND mediatype:{input_data.mediatype}")

        logging.info(f"searching archive.org for: {input_data.query}")

        try:
            response = requests.get(
                "https://archive.org/advancedsearch.php",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            response_data = data.get("response", {})
            results = response_data.get("docs", [])
            total = response_data.get("numFound", 0)

            logging.info(f"found {total} total results, returning {len(results)}")

            return AppOutput(
                results=results,
                total=total,
                raw=data,
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"archive.org api request failed: {e}")
            raise ValueError(f"archive.org search failed: {str(e)}")
