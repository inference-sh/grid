from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional
import os
import requests
import logging
import json

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    query: str = Field(
        description="The question or query to answer",
        min_length=1
    )
    include_full_text: bool = Field(
        default=False,
        description="Include full text content from search results (may cost extra, disabled by default)"
    )
    stream: bool = Field(
        default=False,
        description="Stream the response as server-sent events (note: not fully supported in this implementation)"
    )


class Citation(BaseAppOutput):
    url: str = Field(description="URL of the cited source")
    title: Optional[str] = Field(default=None, description="Title of the cited source")
    author: Optional[str] = Field(default=None, description="Author of the cited source")
    published_date: Optional[str] = Field(default=None, description="Publication date")
    text: Optional[str] = Field(default=None, description="Text content from the source")
    image: Optional[str] = Field(default=None, description="Image URL from the source")
    favicon: Optional[str] = Field(default=None, description="Favicon URL")


class AppOutput(BaseAppOutput):
    answer: str = Field(description="The generated answer based on search results")
    citations: list[dict] = Field(description="Search results that informed the answer")
    formatted_response: str = Field(description="Formatted markdown response with citations")
    cost_dollars: Optional[dict] = Field(default=None, description="API cost breakdown")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Exa answer app."""
        self.api_key = os.environ.get("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable not set")

        self.api_url = "https://api.exa.ai/answer"
        logging.info("Exa Answer app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an answer to the query using Exa API."""
        # Build request payload
        payload = {
            "query": input_data.query,
            "text": input_data.include_full_text,
        }

        # Note: streaming is not fully implemented in this basic version
        # as it requires SSE handling which is complex in this context
        if input_data.stream:
            logging.warning("Streaming mode requested but not fully supported in this implementation")

        # Make API request
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        logging.info(f"Generating answer for query: {input_data.query}")
        logging.info(f"Payload: {payload}")

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=120  # Longer timeout for answer generation
            )
            response.raise_for_status()
            results = response.json()

            # Extract components
            answer = results.get("answer", "")
            citations = results.get("citations", [])
            cost_dollars = results.get("costDollars", None)

            # Format the response with citations
            formatted_response = self._format_answer(answer, citations, input_data.query)

            logging.info(f"Answer generated with {len(citations)} citations")
            if cost_dollars:
                logging.info(f"API cost: ${cost_dollars.get('total', 0):.6f}")

            return AppOutput(
                answer=answer,
                citations=citations,
                formatted_response=formatted_response,
                cost_dollars=cost_dollars
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise ValueError(f"Exa API request failed: {str(e)}")

    def _format_answer(self, answer: str, citations: list[dict], query: str) -> str:
        """Format the answer with proper citations in markdown."""
        output = [f"# Answer to: {query}\n"]

        # Main answer
        output.append("## Answer\n")
        output.append(answer)
        output.append("")

        # Citations section
        if citations:
            output.append("## Sources\n")
            for i, citation in enumerate(citations, 1):
                output.append(f"### [{i}] {citation.get('title', 'No title')}")
                output.append(f"**URL:** {citation.get('url', 'N/A')}")

                if citation.get("author"):
                    output.append(f"**Author:** {citation['author']}")

                if citation.get("publishedDate"):
                    output.append(f"**Published:** {citation['publishedDate']}")

                # Include text excerpt if available
                if citation.get("text"):
                    text = citation["text"]
                    # Show first 300 characters
                    excerpt = text[:300] + "..." if len(text) > 300 else text
                    output.append(f"\n**Excerpt:**\n> {excerpt}")

                output.append("")  # Blank line between citations

        return "\n".join(output)
