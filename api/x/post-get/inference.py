import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    tweet_id: str = Field(description="The ID of the tweet to retrieve")


class AppOutput(BaseAppOutput):
    id: str = Field(description="Post ID")
    text: str = Field(description="Post text content")
    author_id: str = Field(description="Author's user ID")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    like_count: Optional[int] = Field(None, description="Number of likes")
    retweet_count: Optional[int] = Field(None, description="Number of retweets")
    reply_count: Optional[int] = Field(None, description="Number of replies")
    quote_count: Optional[int] = Field(None, description="Number of quotes")
    post_url: str = Field(description="URL of the post")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.posts.get(
                id=input_data.tweet_id,
                tweet_fields=["created_at", "public_metrics", "author_id"]
            )

            data = response.data
            metrics = data.get("public_metrics", {})

            return AppOutput(
                id=data["id"],
                text=data.get("text", ""),
                author_id=data.get("author_id", ""),
                created_at=data.get("created_at"),
                like_count=metrics.get("like_count"),
                retweet_count=metrics.get("retweet_count"),
                reply_count=metrics.get("reply_count"),
                quote_count=metrics.get("quote_count"),
                post_url=f"https://x.com/i/web/status/{data['id']}"
            )
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
