import os
import logging
from typing import Dict, Any, Optional, List, Literal
from tavily import TavilyClient
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field, BaseModel

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Individual search result from Tavily."""
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Content/snippet from the search result")
    score: float = Field(description="Relevance score of the result")
    raw_content: Optional[str] = Field(default=None, description="Raw content if include_raw_content is True")

class AppInput(BaseAppInput):
    """Input schema for Tavily search."""
    query: str = Field(description="Search query to investigate")
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Search depth - 'basic' for quick results, 'advanced' for thorough research"
    )
    topic: Optional[Literal["general", "news"]] = Field(
        default="general",
        description="Search topic category - 'general' for all content, 'news' for recent news"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return (1-20)"
    )
    include_images: bool = Field(
        default=False,
        description="Include images in search results"
    )
    include_answer: bool = Field(
        default=True,
        description="Include AI-generated answer summary"
    )
    include_raw_content: bool = Field(
        default=False,
        description="Include raw HTML content from scraped pages"
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="List of domains to specifically include in search (e.g., ['example.com'])"
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="List of domains to exclude from search (e.g., ['example.com'])"
    )

class AppOutput(BaseAppOutput):
    """Output schema for Tavily search results."""
    answer: Optional[str] = Field(default=None, description="AI-generated answer summary (if include_answer is True)")
    query: str = Field(description="The search query used")
    results: List[SearchResult] = Field(description="List of search results")
    images: Optional[List[str]] = Field(default=None, description="List of image URLs (if include_images is True)")
    response_time: float = Field(description="Response time in seconds")



class App(BaseApp):
    """Tavily Search Assistant App for inference.sh platform."""

    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata=None):
        """Initialize the Tavily search client."""
        logger.info("🔧 Setting up Tavily Search Assistant...")

        # Get API key from environment variable
        api_key = os.environ.get("TVLY_API_KEY")
        if not api_key:
            raise ValueError("TVLY_API_KEY environment variable is not set")

        # Initialize Tavily client
        self.client = TavilyClient(api_key=api_key)
        logger.info("✅ Tavily Search Assistant setup complete")

    async def run(self, input_data: AppInput, metadata=None) -> AppOutput:
        """Execute Tavily search with configurable parameters."""
        try:
            logger.info(f"🔍 Searching with Tavily for: {input_data.query}")
            logger.info(f"    📊 Search depth: {input_data.search_depth}")
            logger.info(f"    📰 Topic: {input_data.topic}")
            logger.info(f"    🔢 Max results: {input_data.max_results}")

            # Build search parameters
            search_params = {
                "query": input_data.query,
                "search_depth": input_data.search_depth,
                "max_results": input_data.max_results,
                "include_images": input_data.include_images,
                "include_answer": input_data.include_answer,
                "include_raw_content": input_data.include_raw_content,
            }

            # Add optional topic parameter
            if input_data.topic:
                search_params["topic"] = input_data.topic

            # Add domain filters if provided
            if input_data.include_domains:
                search_params["include_domains"] = input_data.include_domains
                logger.info(f"    ✅ Including domains: {input_data.include_domains}")

            if input_data.exclude_domains:
                search_params["exclude_domains"] = input_data.exclude_domains
                logger.info(f"    ❌ Excluding domains: {input_data.exclude_domains}")

            # Perform Tavily search
            import time
            start_time = time.time()
            response = self.client.search(**search_params)
            response_time = time.time() - start_time

            logger.info(f"✅ Search completed in {response_time:.2f}s")

            # Parse search results
            results = []
            for result in response.get("results", []):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    raw_content=result.get("raw_content") if input_data.include_raw_content else None
                )
                results.append(search_result)

            logger.info(f"    📄 Found {len(results)} results")

            # Extract answer if provided
            answer = response.get("answer") if input_data.include_answer else None
            if answer:
                logger.info(f"    💡 Generated answer summary")

            # Extract images if requested
            images = response.get("images", []) if input_data.include_images else None
            if images:
                logger.info(f"    🖼️ Found {len(images)} images")

            return AppOutput(
                answer=answer,
                query=input_data.query,
                results=results,
                images=images,
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"❌ Tavily search error: {str(e)}")
            raise ValueError(f"Search failed: {str(e)}")