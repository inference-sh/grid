import os
import logging
from typing import Dict, Any, Optional, List, Literal, Union
from tavily import TavilyClient
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field, BaseModel

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractedContent(BaseModel):
    """Extracted content from a single URL."""
    url: str = Field(description="Source URL that was extracted")
    raw_content: str = Field(description="Extracted content from the webpage")
    images: Optional[List[str]] = Field(default=None, description="List of image URLs found on the page (if include_images is True)")
    favicon: Optional[str] = Field(default=None, description="Favicon URL (if include_favicon is True)")

class FailedExtraction(BaseModel):
    """Information about a failed extraction."""
    url: str = Field(description="URL that failed to extract")
    error: str = Field(description="Error message describing why extraction failed")

class AppInput(BaseAppInput):
    """Input schema for Tavily extract."""
    urls: Union[str, List[str]] = Field(
        description="Single URL or list of URLs to extract content from (max 20 URLs)"
    )
    extract_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Extraction depth - 'basic' for standard content, 'advanced' for tables and embedded content"
    )
    format: Literal["markdown", "text"] = Field(
        default="markdown",
        description="Output format - 'markdown' for formatted content, 'text' for plain text"
    )
    include_images: bool = Field(
        default=False,
        description="Include list of images extracted from the URLs"
    )
    include_favicon: bool = Field(
        default=False,
        description="Include favicon URL for each result"
    )
    timeout: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=60.0,
        description="Request timeout in seconds (1-60). Default: 10s for basic, 30s for advanced"
    )

class AppOutput(BaseAppOutput):
    """Output schema for Tavily extract results."""
    results: List[ExtractedContent] = Field(description="Successfully extracted content from URLs")
    failed_results: List[FailedExtraction] = Field(description="URLs that failed to extract with error messages")
    response_time: float = Field(description="Response time in seconds")
    request_id: str = Field(description="Unique request ID for support reference")
    total_urls: int = Field(description="Total number of URLs processed")
    successful_count: int = Field(description="Number of successfully extracted URLs")
    failed_count: int = Field(description="Number of failed extractions")


class App(BaseApp):
    """Tavily Extract App for inference.sh platform."""

    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata=None):
        """Initialize the Tavily extract client."""
        logger.info("🔧 Setting up Tavily Extract...")

        # Get API key from environment variable
        api_key = os.environ.get("TVLY_API_KEY")
        if not api_key:
            raise ValueError("TVLY_API_KEY environment variable is not set")

        # Initialize Tavily client
        self.client = TavilyClient(api_key=api_key)
        logger.info("✅ Tavily Extract setup complete")

    async def run(self, input_data: AppInput, metadata=None) -> AppOutput:
        """Execute Tavily extract with configurable parameters."""
        try:
            # Normalize URLs to list format
            urls = input_data.urls if isinstance(input_data.urls, list) else [input_data.urls]

            # Validate URL count
            if len(urls) > 20:
                raise ValueError(f"Maximum 20 URLs allowed, got {len(urls)}")

            logger.info(f"🔍 Extracting content from {len(urls)} URL(s)")
            logger.info(f"    📊 Extract depth: {input_data.extract_depth}")
            logger.info(f"    📝 Format: {input_data.format}")

            # Build extract parameters
            extract_params = {
                "urls": urls,
                "extract_depth": input_data.extract_depth,
                "format": input_data.format,
                "include_images": input_data.include_images,
                "include_favicon": input_data.include_favicon,
            }

            # Add optional timeout parameter
            if input_data.timeout:
                extract_params["timeout"] = input_data.timeout
                logger.info(f"    ⏱️ Timeout: {input_data.timeout}s")

            # Perform Tavily extract
            import time
            start_time = time.time()
            response = self.client.extract(**extract_params)
            response_time = time.time() - start_time

            logger.info(f"✅ Extract completed in {response_time:.2f}s")

            # Parse successful results
            results = []
            for result in response.get("results", []):
                extracted = ExtractedContent(
                    url=result.get("url", ""),
                    raw_content=result.get("raw_content", ""),
                    images=result.get("images") if input_data.include_images else None,
                    favicon=result.get("favicon") if input_data.include_favicon else None
                )
                results.append(extracted)

            # Parse failed results
            failed_results = []
            for failed in response.get("failed_results", []):
                failed_extraction = FailedExtraction(
                    url=failed.get("url", ""),
                    error=failed.get("error", "Unknown error")
                )
                failed_results.append(failed_extraction)

            successful_count = len(results)
            failed_count = len(failed_results)
            total_urls = successful_count + failed_count

            logger.info(f"    ✅ Successfully extracted: {successful_count}/{total_urls}")
            if failed_count > 0:
                logger.warning(f"    ❌ Failed extractions: {failed_count}/{total_urls}")

            # Extract request_id for support reference
            request_id = response.get("request_id", "")

            return AppOutput(
                results=results,
                failed_results=failed_results,
                response_time=response_time,
                request_id=request_id,
                total_urls=total_urls,
                successful_count=successful_count,
                failed_count=failed_count
            )

        except Exception as e:
            logger.error(f"❌ Tavily extract error: {str(e)}")
            raise ValueError(f"Extract failed: {str(e)}")
