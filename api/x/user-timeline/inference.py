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
    """Input schema for home timeline."""
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum number of results (1-100)")
    exclude_replies: bool = Field(default=False, description="Exclude replies from timeline")
    exclude_retweets: bool = Field(default=False, description="Exclude retweets from timeline")


class AppOutput(BaseAppOutput):
    """Output schema for home timeline."""
    posts: List[PostInfo] = Field(description="List of timeline posts")


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

            print(f"Fetching timeline for user {user_id}")

            exclude = []
            if input_data.exclude_replies:
                exclude.append("replies")
            if input_data.exclude_retweets:
                exclude.append("retweets")

            kwargs = dict(
                id=user_id,
                max_results=input_data.max_results,
                tweet_fields=["id", "text", "author_id", "created_at", "public_metrics", "article", "article_title"],
                expansions=["author_id"],
                user_fields=["id", "name", "username", "profile_image_url"],
            )
            if exclude:
                kwargs["exclude"] = exclude

            posts = []
            users_map = {}

            for page in self.client.users.get_timeline(**kwargs):
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

            print(f"Found {len(posts)} timeline posts")

            return AppOutput(posts=posts)

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
