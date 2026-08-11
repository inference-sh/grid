import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
from .x_helper import raise_api_error


class PostInfo(BaseAppInput):
    """A bookmarked post."""
    id: str = Field(description="Post ID")
    text: str = Field(description="Post text content")
    author_id: str = Field(description="Author's user ID")
    author_name: Optional[str] = Field(None, description="Author display name")
    author_username: Optional[str] = Field(None, description="Author @username")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    like_count: Optional[int] = Field(None, description="Number of likes")
    retweet_count: Optional[int] = Field(None, description="Number of retweets")
    reply_count: Optional[int] = Field(None, description="Number of replies")
    post_url: Optional[str] = Field(None, description="URL of the post")
    has_article: Optional[bool] = Field(None, description="Whether the post has an article")


class AppInput(BaseAppInput):
    """Input schema for listing bookmarks."""
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of bookmarks to return"
    )


class AppOutput(BaseAppOutput):
    """Output schema for bookmarks list."""
    posts: List[PostInfo] = Field(description="List of bookmarked posts")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            me = self.client.users.get_me()
            user_id = me.data.id if hasattr(me.data, 'id') else me.data['id']

            posts = []
            users_map = {}

            for page in self.client.users.get_bookmarks(
                id=user_id,
                max_results=input_data.max_results,
                tweet_fields=["id", "text", "author_id", "created_at", "public_metrics", "article", "article_title"],
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
                    tweet_id = str(td.get("id", ""))
                    author_username = user_info.get("username")

                    post_url = f"https://x.com/{author_username}/status/{tweet_id}" if author_username else f"https://x.com/i/web/status/{tweet_id}"

                    has_article = bool(td.get("article") or td.get("article_title"))

                    posts.append(PostInfo(
                        id=tweet_id,
                        text=td.get("text", ""),
                        author_id=author_id,
                        author_name=user_info.get("name"),
                        author_username=author_username,
                        created_at=td.get("created_at"),
                        like_count=metrics.get("like_count"),
                        retweet_count=metrics.get("retweet_count"),
                        reply_count=metrics.get("reply_count"),
                        post_url=post_url,
                        has_article=has_article,
                    ))

                break

            return AppOutput(posts=posts)

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
