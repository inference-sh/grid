import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional
from .x_helper import raise_api_error


class TrendInfo(BaseAppInput):
    """A trending topic."""
    name: str = Field(description="Trend name or hashtag")
    tweet_count: Optional[int] = Field(None, description="Number of tweets for this trend")
    url: Optional[str] = Field(None, description="URL to view this trend on X")


class AppInput(BaseAppInput):
    """Input schema for getting trends."""
    woeid: int = Field(
        default=1,
        description="Where On Earth ID. Use 1 for worldwide, or a specific location ID "
        "(e.g. 23424977 for US, 23424975 for UK, 23424856 for Japan)"
    )


class AppOutput(BaseAppOutput):
    """Output schema for trends."""
    woeid: int = Field(description="The WOEID queried")
    trends: List[TrendInfo] = Field(description="Trending topics")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.trends.get_by_woeid(woeid=input_data.woeid)

            trends = []
            data = getattr(response, "data", None) or response

            # Response may be a list or have a trends attribute
            trend_list = data
            if hasattr(data, "trends"):
                trend_list = data.trends
            elif not isinstance(data, list):
                trend_list = [data] if data else []

            for trend in trend_list:
                td = trend.model_dump() if hasattr(trend, "model_dump") else trend
                if isinstance(td, dict):
                    trends.append(TrendInfo(
                        name=td.get("name", ""),
                        tweet_count=td.get("tweet_count") or td.get("tweet_volume"),
                        url=td.get("url"),
                    ))

            return AppOutput(
                woeid=input_data.woeid,
                trends=trends,
            )

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
