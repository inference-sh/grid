import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
from .x_helper import raise_api_error


class PostInfo(BaseAppInput):
    """A quote tweet."""
    id: str = Field(description="Post ID")
    text: str = Field(description="Post text content")
    author_id: str = Field(description="Author's user ID")
    author_name: Optional[str] = Field(None, description="Author display name")
    author_username: Optional[str] = Field(None, description="Author @username")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    like_count: Optional[int] = Field(None, description="Number of likes")
    retweet_count: Optional[int] = Field(None, description="Number of retweets")
    post_url: Optional[str] = Field(None, description="URL of the post")


class AppInput(BaseAppInput):
    """Input schema for listing quote tweets."""
    tweet_id: str = Field(description="The ID of the post to get quotes for")
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of quotes to return"
    )


class AppOutput(BaseAppOutput):
    """Output schema for quote tweets list."""
    tweet_id: str = Field(description="The original post ID")
    quotes: List[PostInfo] = Field(description="List of quote tweets")
    count: int = Field(description="Number of quote tweets returned")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            quotes = []
            users_map = {}

            for page in self.client.posts.get_quoted(
                id=input_data.tweet_id,
                max_results=input_data.max_results,
                tweet_fields=["id", "text", "author_id", "created_at", "public_metrics"],
                expansions=["author_id"],
                user_fields=["id", "name", "username"],
            ):
                includes = getattr(page, "includes", None)
                if includes:
                    for u in (getattr(includes, "users", None) or []):
                        ud = u.model_dump() if hasattr(u, 'model_dump') else u
                        users_map[str(ud.get("id", ""))] = ud

                data = getattr(page, "data", None) or []
                for tweet in data:
                    td = tweet.model_dump() if hasattr(tweet, 'model_dump') else tweet
                    metrics = td.get("public_metrics", {}) or {}
                    author_id = str(td.get("author_id", ""))
                    user_info = users_map.get(author_id, {})
                    quote_id = str(td.get("id", ""))
                    author_username = user_info.get("username")

                    post_url = f"https://x.com/{author_username}/status/{quote_id}" if author_username else f"https://x.com/i/web/status/{quote_id}"

                    quotes.append(PostInfo(
                        id=quote_id,
                        text=td.get("text", ""),
                        author_id=author_id,
                        author_name=user_info.get("name"),
                        author_username=author_username,
                        created_at=td.get("created_at"),
                        like_count=metrics.get("like_count"),
                        retweet_count=metrics.get("retweet_count"),
                        post_url=post_url,
                    ))

                break

            return AppOutput(
                tweet_id=input_data.tweet_id,
                quotes=quotes,
                count=len(quotes),
            )

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
