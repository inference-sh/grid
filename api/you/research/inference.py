"""
You.com Research API.

Multi-step web research with source-backed citations.
Configurable effort levels: lite, standard, deep, exhaustive.
"""

from enum import Enum
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, RawMeta
from pydantic import Field, BaseModel
from typing import Optional, List

from .you_helper import get_api_key, setup_logger, research_request


class ResearchEffortEnum(str, Enum):
    lite = "lite"
    standard = "standard"
    deep = "deep"
    exhaustive = "exhaustive"


class Source(BaseModel):
    title: str = Field(default="", description="Source title")
    url: str = Field(default="", description="Source URL")


class AppInput(BaseAppInput):
    query: str = Field(description="Research question (up to 40,000 characters)", max_length=40000)
    research_effort: ResearchEffortEnum = Field(
        default=ResearchEffortEnum.standard,
        description="Research depth: lite (fast), standard (balanced), deep (thorough), exhaustive (comprehensive)"
    )


class AppOutput(BaseAppOutput):
    answer: str = Field(description="Synthesized research answer with citations")
    sources: List[Source] = Field(default_factory=list, description="Sources used in the research")
    raw: dict = Field(description="Full API response")


# Cost per call in cents by effort level
EFFORT_COST_CENTS = {
    "lite": 0.65,       # $6.50/1000
    "standard": 5.0,    # $50/1000
    "deep": 10.0,       # $100/1000
    "exhaustive": 30.0, # $300/1000
}


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("You.com Research initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = research_request(
                api_key=self.api_key,
                query=input_data.query,
                research_effort=input_data.research_effort.value,
                logger=self.logger,
            )

            output = response.get("output", {})
            answer = output.get("content", "")
            raw_sources = output.get("sources", [])

            sources = []
            for src in raw_sources:
                sources.append(Source(
                    title=src.get("title", ""),
                    url=src.get("url", ""),
                ))

            self.logger.info(f"Research complete: {len(sources)} sources, {len(answer)} chars")

            cost_cents = EFFORT_COST_CENTS.get(input_data.research_effort.value, 5.0)

            return AppOutput(
                answer=answer,
                sources=sources,
                raw=response,
                output_meta=OutputMeta(outputs=[RawMeta(cost=cost_cents)]),
            )

        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            raise RuntimeError(f"Research failed: {str(e)}")
