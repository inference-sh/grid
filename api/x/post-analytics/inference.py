import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List
from .x_helper import raise_api_error


class AppInput(BaseAppInput):
    """Input schema for post analytics."""
    tweet_ids: List[str] = Field(
        description="List of post IDs to get analytics for"
    )
    start_time: str = Field(
        description="Start time in ISO 8601 format (e.g. 2024-01-01T00:00:00Z)"
    )
    end_time: str = Field(
        description="End time in ISO 8601 format (e.g. 2024-01-31T23:59:59Z)"
    )
    granularity: str = Field(
        default="day",
        description="Time granularity: 'day', 'hour', or 'total'"
    )


class AppOutput(BaseAppOutput):
    """Output schema for post analytics."""
    analytics: List[dict] = Field(description="Analytics data for the requested posts")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.posts.get_analytics(
                ids=input_data.tweet_ids,
                start_time=input_data.start_time,
                end_time=input_data.end_time,
                granularity=input_data.granularity,
            )

            # Not paginated — single response
            data = getattr(response, "data", None) or response
            if hasattr(data, "model_dump"):
                analytics = data.model_dump()
                if isinstance(analytics, dict):
                    analytics = [analytics]
            elif isinstance(data, list):
                analytics = [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in data
                ]
            else:
                analytics = [data] if data else []

            return AppOutput(analytics=analytics)

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
