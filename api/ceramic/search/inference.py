import os
import logging
import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta
from pydantic import Field
from typing import Optional

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    query: str = Field(
        description="Search query in keyword format (1-50 words)",
        default="latest developments in artificial intelligence",
    )
    max_description_length: int = Field(
        default=3000,
        ge=1000,
        le=8000,
        description="Maximum character length for each result's description",
    )


class AppOutput(BaseAppOutput):
    results: list[dict] = Field(description="Search result items")
    total_results: int = Field(description="Total number of results returned")
    execution_time: float = Field(description="Search execution time in seconds")
    answer: str = Field(description="Formatted summary of search results")


class App(BaseApp):
    async def setup(self, metadata):
        self.api_key = os.environ.get("CERAMIC_KEY")
        if not self.api_key:
            raise ValueError("CERAMIC_KEY environment variable not set")
        self.client = httpx.AsyncClient(timeout=60)
        logging.info("ceramic search app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        payload = {
            "query": input_data.query,
            "maxDescriptionLength": input_data.max_description_length,
        }

        logging.info(f"searching ceramic for: {input_data.query}")

        response = await self.client.post(
            "https://api.ceramic.ai/search",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if response.status_code not in (200, 201):
            logging.error(f"ceramic API error {response.status_code}: {response.text}")
            response.raise_for_status()

        data = response.json()
        result = data.get("result", {})
        items = result.get("results", [])
        total = result.get("totalResults", len(items))
        exec_time = result.get("searchMetadata", {}).get("executionTime", 0)

        logging.info(f"search returned {len(items)} results in {exec_time:.3f}s")

        answer = self._format_results(items, input_data.query, total)

        return AppOutput(
            results=items,
            total_results=total,
            execution_time=exec_time,
            answer=answer,
            output_meta=OutputMeta(),
        )

    def _format_results(self, items: list, query: str, total: int) -> str:
        lines = [f"# search results for: {query}\n", f"**{total} results found**\n"]
        for i, item in enumerate(items, 1):
            lines.append(f"\n## {i}. {item.get('title', 'no title')}")
            lines.append(f"**url:** {item.get('url', 'n/a')}")
            desc = item.get("description", "")
            if desc:
                snippet = desc[:500] + "..." if len(desc) > 500 else desc
                lines.append(f"\n{snippet}")
        return "\n".join(lines)

    async def unload(self):
        await self.client.aclose()
