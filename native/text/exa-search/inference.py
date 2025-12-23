from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, Literal
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    query: str = Field(
        description="The search query string",
        default="Latest developments in LLM capabilities"
    )
    search_type: Literal["auto", "neural", "keyword"] = Field(
        default="auto",
        description="Search method: auto (default), neural (embeddings), or keyword"
    )
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of search results to return (max 100)"
    )
    category: Optional[Literal[
        "company", "research paper", "news", "pdf", "github",
        "tweet", "personal site", "linkedin profile", "financial report"
    ]] = Field(
        default=None,
        description="Focus search on specific content category"
    )
    include_domains: Optional[list[str]] = Field(
        default=None,
        description="List of domains to restrict search to (e.g., ['arxiv.org', 'github.com'])"
    )
    exclude_domains: Optional[list[str]] = Field(
        default=None,
        description="List of domains to exclude from search"
    )
    start_published_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 datetime - only content published after this date"
    )
    end_published_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 datetime - only content published before this date"
    )
    include_text: Optional[list[str]] = Field(
        default=None,
        description="Required text strings that must appear in results (max 5 words per string)"
    )
    exclude_text: Optional[list[str]] = Field(
        default=None,
        description="Text strings that must NOT appear in results"
    )
    get_contents: bool = Field(
        default=True,
        description="Include full text content in results"
    )
    get_summary: bool = Field(
        default=False,
        description="Generate LLM-powered summaries for each result (costs extra)"
    )


class AppOutput(BaseAppOutput):
    results: dict = Field(description="Complete search results from Exa API")
    answer: str = Field(description="Formatted summary of search results")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Exa search app."""
        self.api_key = os.environ.get("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable not set")

        self.api_url = "https://api.exa.ai/search"
        logging.info("Exa Search app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Execute Exa search with the provided parameters."""
        # Build request payload
        payload = {
            "query": input_data.query,
            "type": input_data.search_type,
            "numResults": input_data.num_results,
        }

        # Add optional parameters
        if input_data.category:
            payload["category"] = input_data.category

        if input_data.include_domains:
            payload["includeDomains"] = input_data.include_domains

        if input_data.exclude_domains:
            payload["excludeDomains"] = input_data.exclude_domains

        if input_data.start_published_date:
            payload["startPublishedDate"] = input_data.start_published_date

        if input_data.end_published_date:
            payload["endPublishedDate"] = input_data.end_published_date

        if input_data.include_text:
            payload["includeText"] = input_data.include_text

        if input_data.exclude_text:
            payload["excludeText"] = input_data.exclude_text

        # Configure content retrieval (no highlights to avoid costs)
        contents_config = {}

        if input_data.get_contents:
            contents_config["text"] = True

        if input_data.get_summary:
            contents_config["summary"] = True

        if contents_config:
            payload["contents"] = contents_config

        # Make API request
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        logging.info(f"Searching Exa for: {input_data.query}")
        logging.info(f"Payload: {payload}")

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            results = response.json()

            # Format answer summary
            answer = self._format_results(results, input_data.query)

            logging.info(f"Search completed with {len(results.get('results', []))} results")

            return AppOutput(
                results=results,
                answer=answer
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise ValueError(f"Exa API request failed: {str(e)}")

    def _format_results(self, results: dict, query: str) -> str:
        """Format search results into a readable summary."""
        output = [f"# Search Results for: {query}\n"]

        search_type = results.get("resolvedSearchType", "unknown")
        output.append(f"**Search Type:** {search_type}\n")

        result_items = results.get("results", [])
        output.append(f"**Found {len(result_items)} results**\n")

        for i, result in enumerate(result_items, 1):
            output.append(f"\n## {i}. {result.get('title', 'No title')}")
            output.append(f"**URL:** {result.get('url', 'N/A')}")

            if result.get("author"):
                output.append(f"**Author:** {result['author']}")

            if result.get("publishedDate"):
                output.append(f"**Published:** {result['publishedDate']}")

            if result.get("summary"):
                output.append(f"\n**Summary:** {result['summary']}")

            if result.get("text"):
                # Show snippet of text
                text_snippet = result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
                output.append(f"\n**Excerpt:** {text_snippet}")

        # Add cost information if available
        if "costDollars" in results:
            cost = results["costDollars"].get("total", 0)
            output.append(f"\n\n**API Cost:** ${cost:.6f}")

        return "\n".join(output)
