import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field


class AppInput(BaseAppInput):
    tweet_id: str = Field(description="The ID of the tweet to retweet")


class AppOutput(BaseAppOutput):
    retweeted: bool = Field(description="Whether the post was successfully retweeted")
    tweet_id: str = Field(description="ID of the retweeted tweet")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.reposts.create(tweet_id=input_data.tweet_id)
            retweeted = getattr(response.data, "retweeted", True)

            return AppOutput(retweeted=retweeted, tweet_id=input_data.tweet_id)
        except Exception as e:
            error_msg = str(e).lower()
            if "already retweeted" in error_msg:
                return AppOutput(retweeted=True, tweet_id=input_data.tweet_id)
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
