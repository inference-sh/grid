from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, Literal
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    url: str = Field(
        description="URL to extract content from (single URL for cost predictability)",
        min_length=1
    )
    get_text: bool = Field(
        default=True,
        description="Extract full page text content"
    )
    max_characters: Optional[int] = Field(
        default=None,
        description="Maximum characters to extract from the page (default: no limit)"
    )
    include_html_tags: bool = Field(
        default=False,
        description="Preserve HTML structure in extracted text"
    )
    get_summary: bool = Field(
        default=False,
        description="Generate LLM-powered summary (costs extra)"
    )
    summary_query: Optional[str] = Field(
        default=None,
        description="Custom query to guide summary generation"
    )
    livecrawl: Literal["never", "fallback", "always", "preferred"] = Field(
        default="fallback",
        description="Page crawling strategy: never, fallback (default), always, or preferred"
    )
    livecrawl_timeout: int = Field(
        default=10000,
        ge=1000,
        le=60000,
        description="Crawl timeout in milliseconds (default: 10000ms)"
    )
    subpages: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of subpages to crawl per URL"
    )
    subpage_target: Optional[str] = Field(
        default=None,
        description="Keywords to target specific subpages"
    )
    extract_links: int = Field(
        default=0,
        ge=0,
        description="Number of links to extract per page"
    )
    extract_image_links: int = Field(
        default=0,
        ge=0,
        description="Number of image URLs to extract per page"
    )
    create_context: bool = Field(
        default=False,
        description="Combine all contents into single LLM-ready context string"
    )


class AppOutput(BaseAppOutput):
    results: dict = Field(description="Complete extraction results from Exa API")
    answer: str = Field(description="Formatted summary of extracted content")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Exa extract app."""
        self.api_key = os.environ.get("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable not set")

        self.api_url = "https://api.exa.ai/contents"
        logging.info("Exa Extract app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Extract content from URL using Exa API."""
        # Build request payload (single URL for cost predictability)
        payload = {
            "urls": [input_data.url],  # API expects a list
            "livecrawl": input_data.livecrawl,
            "livecrawlTimeout": input_data.livecrawl_timeout,
        }

        # Configure text extraction
        if input_data.get_text:
            text_config = {
                "includeHtmlTags": input_data.include_html_tags
            }
            if input_data.max_characters:
                text_config["maxCharacters"] = input_data.max_characters
            payload["text"] = text_config

        # Configure summary (no highlights to avoid costs)
        if input_data.get_summary:
            summary_config = {}
            if input_data.summary_query:
                summary_config["query"] = input_data.summary_query
            payload["summary"] = summary_config

        # Configure subpages
        if input_data.subpages > 0:
            payload["subpages"] = input_data.subpages
            if input_data.subpage_target:
                payload["subpageTarget"] = input_data.subpage_target

        # Configure extras (links and images)
        if input_data.extract_links > 0 or input_data.extract_image_links > 0:
            extras_config = {}
            if input_data.extract_links > 0:
                extras_config["links"] = input_data.extract_links
            if input_data.extract_image_links > 0:
                extras_config["imageLinks"] = input_data.extract_image_links
            payload["extras"] = extras_config

        # Configure context
        if input_data.create_context:
            payload["context"] = True

        # Make API request
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        logging.info(f"Extracting content from URL: {input_data.url}")
        logging.info(f"Payload: {payload}")

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=120  # Longer timeout for content extraction
            )
            response.raise_for_status()
            results = response.json()

            # Format answer summary
            answer = self._format_results(results, input_data.url)

            logging.info(f"Extraction completed successfully")

            return AppOutput(
                results=results,
                answer=answer
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise ValueError(f"Exa API request failed: {str(e)}")

    def _format_results(self, results: dict, url: str) -> str:
        """Format extraction results into a readable summary."""
        output = [f"# Content Extraction Results\n"]
        output.append(f"**URL:** {url}\n")

        # Check for statuses
        statuses = results.get("statuses", [])
        if statuses:
            failed = [s for s in statuses if s.get("status") != "success"]
            if failed:
                output.append(f"**Warning:** Extraction failed\n")

        result_items = results.get("results", [])

        if result_items:
            result = result_items[0]  # Single result
            output.append(f"\n## {result.get('title', 'No title')}")

            if result.get("author"):
                output.append(f"**Author:** {result['author']}")

            if result.get("publishedDate"):
                output.append(f"**Published:** {result['publishedDate']}")

            # Summary
            if result.get("summary"):
                output.append(f"\n**Summary:**\n{result['summary']}")

            # Text content (show snippet)
            if result.get("text"):
                text_length = len(result["text"])
                output.append(f"\n**Content:** {text_length} characters")
                if text_length > 0:
                    snippet = result["text"][:1000] + "..." if text_length > 1000 else result["text"]
                    output.append(f"```\n{snippet}\n```")

            # Subpages
            if result.get("subpages"):
                output.append(f"\n**Subpages:** {len(result['subpages'])} found")
                for j, subpage in enumerate(result["subpages"][:3], 1):
                    output.append(f"  {j}. {subpage.get('title', 'No title')} - {subpage.get('url', 'N/A')}")

            # Extras (links, images)
            if result.get("extras"):
                extras = result["extras"]
                if extras.get("links"):
                    output.append(f"\n**Extracted Links:** {len(extras['links'])}")
                if extras.get("imageLinks"):
                    output.append(f"**Extracted Images:** {len(extras['imageLinks'])}")

        # Context string
        if results.get("context"):
            context_length = len(results["context"])
            output.append(f"\n\n**Combined Context:** {context_length} characters")
            output.append("(Use this for LLM processing)")

        # Status details
        if statuses:
            output.append("\n\n## Extraction Status")
            for i, status in enumerate(statuses, 1):
                status_str = status.get("status", "unknown")
                output.append(f"{i}. {status_str}")
                if status.get("error"):
                    output.append(f"   Error: {status['error']}")

        # Cost information
        if "costDollars" in results:
            cost = results["costDollars"].get("total", 0)
            output.append(f"\n**API Cost:** ${cost:.6f}")

        return "\n".join(output)
