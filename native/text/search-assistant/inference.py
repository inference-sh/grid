import os
import logging
from typing import Dict, Any, Optional, List, Literal
from openai import OpenAI
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field, BaseModel

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchSource(BaseModel):
    """Information about a research source."""
    type: str = Field(description="Type of source (ai_research, web_search, etc.)")
    model: Optional[str] = Field(default=None, description="Model used for research")
    query: str = Field(description="Query that generated this source")
    url: Optional[str] = Field(default=None, description="Source URL if available")
    title: Optional[str] = Field(default=None, description="Source title if available")

class AppInput(BaseAppInput):
    """Input schema for the research assistant."""
    query: str = Field(description="Research query or question to investigate")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt for research context")
    model: Literal[
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "o1-mini",
        "o1-preview",
        "o3-mini",
        "chatgpt-4o-latest"
    ] = Field(default="gpt-4o-mini", description="Model to use for research")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for response randomness (0.0-2.0)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response (optional)")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter (0.0-1.0)")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty (-2.0 to 2.0)")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty (-2.0 to 2.0)")
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = Field(default=None, description="Reasoning effort level for research analysis (optional, only for supported models)")

class AppOutput(BaseAppOutput):
    """Output schema for the research assistant."""
    results: str = Field(description="Research findings and analysis")


class App(BaseApp):
    """Research Assistant App for inference.sh platform."""

    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata=None):
        """Initialize the research assistant."""
        logger.info("🔧 Setting up Research Assistant...")
        logger.info("✅ Research Assistant setup complete")

    async def run(self, input_data: AppInput, metadata=None) -> AppOutput:
        """Execute research query with configurable parameters."""
        try:
            logger.info(f"🔍 Conducting research for: {input_data.query}")

            # Get API key from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            # Initialize client with API key from environment
            self.client = OpenAI(api_key=api_key)

            # Perform research using flattened parameters
            research_result = await self._perform_research(
                query=input_data.query,
                model=input_data.model,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
                top_p=input_data.top_p,
                frequency_penalty=input_data.frequency_penalty,
                presence_penalty=input_data.presence_penalty,
                reasoning_effort=input_data.reasoning_effort,
                custom_system_prompt=input_data.system_prompt
            )

            if research_result["status"] == "success":
                logger.info(f"✅ Research completed successfully")

                return AppOutput(
                    results=research_result["text"]
                )
            else:
                error_msg = research_result.get("error", "Unknown error during research")
                logger.error(f"❌ Research failed: {error_msg}")
                raise ValueError(f"Research failed: {error_msg}")

        except Exception as e:
            logger.error(f"❌ Research error: {str(e)}")
            raise ValueError(f"Research failed: {str(e)}")

    async def _perform_research(
        self,
        query: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        reasoning_effort: Optional[str],
        custom_system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform research using the language model."""
        try:
            # Default system prompt for research
            default_system_prompt = (
                "You are a comprehensive research specialist with expertise across multiple domains. "
                "Provide detailed, accurate, and well-structured information with specific facts, "
                "statistics, data points, and evidence-based insights. Focus on being thorough, "
                "informative, and objective in your analysis. Include relevant context, "
                "background information, and practical implications when appropriate."
            )

            system_prompt = custom_system_prompt or default_system_prompt

            logger.info(f"    🤖 Using model: {model}")
            logger.info(f"    🌡️ Temperature: {temperature}")
            if reasoning_effort:
                logger.info(f"    🧠 Reasoning effort: {reasoning_effort}")
            else:
                logger.info(f"    🧠 Reasoning: Not specified (using standard completion)")

            # Prepare the input prompt with system context
            full_prompt = f"{system_prompt}\n\nUser request: Research and provide comprehensive information about: {query}"

            # Make API call - use responses.create if reasoning is specified, otherwise use chat.completions.create
            if reasoning_effort:
                # Prepare API call parameters for responses.create (reasoning models)
                # Note: responses.create API doesn't support max_tokens, temperature, top_p, etc.
                api_params = {
                    "model": model,
                    "input": full_prompt,
                    "reasoning": {"effort": reasoning_effort}
                }
                response = self.client.responses.create(**api_params)
            else:
                # Fallback to standard chat completions API for models without reasoning support
                chat_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Research and provide comprehensive information about: {query}"}
                    ]
                }

                # Add optional parameters for chat completions
                if max_tokens:
                    chat_params["max_tokens"] = max_tokens
                if temperature != 0.1:
                    chat_params["temperature"] = temperature
                if top_p != 1.0:
                    chat_params["top_p"] = top_p
                if frequency_penalty != 0.0:
                    chat_params["frequency_penalty"] = frequency_penalty
                if presence_penalty != 0.0:
                    chat_params["presence_penalty"] = presence_penalty

                response = self.client.chat.completions.create(**chat_params)

            # Extract response content and usage - handle both API formats
            content = None
            tokens_used = None

            if reasoning_effort:
                # Handle responses.create format
                if hasattr(response, 'output_text') and response.output_text:
                    content = response.output_text
                    tokens_used = getattr(response, 'usage', {}).get('total_tokens') if hasattr(response, 'usage') else None
            else:
                # Handle chat.completions.create format
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens if response.usage else None

            if content:
                # Create source information
                sources = [ResearchSource(
                    type="ai_research",
                    model=model,
                    query=query
                )]

                return {
                    "text": content,
                    "sources": sources,
                    "status": "success",
                    "error": None,
                    "tokens_used": tokens_used,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "reasoning_effort": reasoning_effort,
                    "custom_system_prompt": custom_system_prompt,
                }
            else:
                raise ValueError("No response received from language model")

        except Exception as e:
            logger.error(f"    ❌ Research API error: {str(e)}")
            raise ValueError(f"Research API failed: {str(e)}")