from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

STRUCTURED_PROMPT_BASE = "https://engine.prod.bria-api.com/v2/structured_prompt"


class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(default=None, description="Text prompt to convert to structured format")
    images: Optional[list[File]] = Field(default=None, description="Reference images to analyze")


class AppOutput(BaseAppOutput):
    structured_prompt: str = Field(description="Structured prompt JSON")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Structured Prompt ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {}
        if input_data.prompt is not None:
            payload["prompt"] = input_data.prompt
        if input_data.images:
            payload["image_urls"] = [img.uri for img in input_data.images]

        logger.info("Requesting structured prompt generation")
        result = await bria_helper.call_endpoint(
            self.client, "generate", payload, base_url=STRUCTURED_PROMPT_BASE
        )

        structured_prompt = result["result"]["structured_prompt"]

        return AppOutput(structured_prompt=structured_prompt)

    async def unload(self):
        await self.client.aclose()
