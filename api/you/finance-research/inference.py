"""
You.com Finance Research API.

Agentic financial research with SEC filings, macro data, and institutional-grade sources.
"""

from enum import Enum
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, RawMeta
from pydantic import Field, BaseModel
from typing import List

from .you_helper import get_api_key, setup_logger, finance_research_request


class FinanceEffortEnum(str, Enum):
    deep = "deep"
    exhaustive = "exhaustive"


class Source(BaseModel):
    title: str = Field(default="", description="Source title")
    url: str = Field(default="", description="Source URL")


class AppInput(BaseAppInput):
    query: str = Field(description="Financial research question", max_length=40000)
    research_effort: FinanceEffortEnum = Field(
        default=FinanceEffortEnum.deep,
        description="Research depth: deep (thorough), exhaustive (comprehensive)"
    )


class AppOutput(BaseAppOutput):
    answer: str = Field(description="Synthesized financial research with inline citations")
    sources: List[Source] = Field(default_factory=list, description="Sources used")
    raw: dict = Field(description="Full API response")


EFFORT_COST_CENTS = {
    "deep": 11.0,       # $110/1000
    "exhaustive": 25.0, # estimated
}


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("You.com Finance Research initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = finance_research_request(
                api_key=self.api_key,
                query=input_data.query,
                research_effort=input_data.research_effort.value,
                logger=self.logger,
            )

            output = response.get("output", {})
            answer = output.get("content", "")
            raw_sources = output.get("sources", [])

            sources = [Source(title=s.get("title", ""), url=s.get("url", "")) for s in raw_sources]

            self.logger.info(f"Finance research complete: {len(sources)} sources, {len(answer)} chars")

            cost_cents = EFFORT_COST_CENTS.get(input_data.research_effort.value, 11.0)

            return AppOutput(
                answer=answer,
                sources=sources,
                raw=response,
                output_meta=OutputMeta(outputs=[RawMeta(cost=cost_cents)]),
            )

        except Exception as e:
            self.logger.error(f"Finance research failed: {e}")
            raise RuntimeError(f"Finance research failed: {str(e)}")
