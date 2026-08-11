import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
from .x_helper import raise_api_error


class PostInfo(BaseAppInput):
    """A single post."""
    id: str = Field(description="Post ID")
    text: str = Field(description="Post text content")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    like_count: Optional[int] = Field(None, description="Number of likes")
    retweet_count: Optional[int] = Field(None, description="Number of retweets")
    reply_count: Optional[int] = Field(None, description="Number of replies")
    post_url: Optional[str] = Field(None, description="URL to the post")
    has_article: bool = Field(False, description="Whether the post has an attached article")


class AppInput(BaseAppInput):
    """Input schema for listing user posts."""
    username: Optional[str] = Field(None, description="Username to look up (with or without @)")
    user_id: Optional[str] = Field(None, description="User ID to look up")
    max_results: int = Field(default=20, ge=5, le=100, description="Maximum number of results (5-100)")


class AppOutput(BaseAppOutput):
    """Output schema for user posts."""
    user_id: str = Field(description="User ID whose posts were fetched")
    posts: List[PostInfo] = Field(description="List of posts")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        if not input_data.user_id and not input_data.username:
            raise ValueError("Either user_id or username must be provided")

        try:
            user_id = input_data.user_id

            if not user_id:
                resp = self.client.users.get_by_username(username=input_data.username.lstrip("@"))
                user_id = resp.data.id

            print(f"Fetching posts for user {user_id}")

            posts = []
            users_map = {}

            for page in self.client.users.get_posts(
                id=user_id,
                max_results=input_data.max_results,
                tweet_fields=["id", "text", "author_id", "created_at", "public_metrics", "article", "article_title"],
                expansions=["author_id"],
                user_fields=["id", "name", "username", "profile_image_url"],
            ):
                includes = getattr(page, "includes", None)
                if includes:
                    page_users = getattr(includes, "users", None) or []
                    for u in page_users:
                        ud = u.model_dump() if hasattr(u, 'model_dump') else (u if isinstance(u, dict) else {})
                        uid = str(ud.get("id", ""))
                        if uid:
                            users_map[uid] = ud.get("username")

                data = getattr(page, "data", None) or []
                for tweet in data:
                    td = tweet.model_dump() if hasattr(tweet, 'model_dump') else (tweet if isinstance(tweet, dict) else {})
                    metrics = td.get("public_metrics", {}) or {}
                    author_id = str(td.get("author_id", ""))
                    author_username = users_map.get(author_id)
                    tweet_id = str(td.get("id", ""))

                    post_url = None
                    if author_username and tweet_id:
                        post_url = f"https://x.com/{author_username}/status/{tweet_id}"
                    elif tweet_id:
                        post_url = f"https://x.com/i/web/status/{tweet_id}"

                    has_article = bool(td.get("article") or td.get("article_title"))

                    posts.append(PostInfo(
                        id=tweet_id,
                        text=td.get("text", ""),
                        created_at=td.get("created_at"),
                        like_count=metrics.get("like_count"),
                        retweet_count=metrics.get("retweet_count"),
                        reply_count=metrics.get("reply_count"),
                        post_url=post_url,
                        has_article=has_article,
                    ))

                    if len(posts) >= input_data.max_results:
                        break

                break  # first page only

            print(f"Found {len(posts)} posts")

            return AppOutput(
                user_id=str(user_id),
                posts=posts,
            )

        except ValueError:
            raise
        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
