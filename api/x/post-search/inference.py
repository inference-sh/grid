import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
from .x_helper import raise_api_error


class AppInput(BaseAppInput):
    """Input schema for searching posts."""
    query: str = Field(
        description="X search query. Use 'conversation_id:TWEET_ID' to get replies to a tweet, "
        "or any standard X search query (keywords, from:user, to:user, etc.)"
    )
    max_results: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Maximum number of results to return (10-100)"
    )
    sort_order: Optional[str] = Field(
        default="relevancy",
        description="Sort order: 'relevancy' or 'recency'"
    )


class Post(BaseAppInput):
    """A single post in the search results."""
    id: str = Field(description="Post ID")
    text: str = Field(description="Post text content")
    author_id: str = Field(description="Author's user ID")
    author_name: Optional[str] = Field(None, description="Author display name")
    author_username: Optional[str] = Field(None, description="Author @username")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    like_count: Optional[int] = Field(None, description="Number of likes")
    retweet_count: Optional[int] = Field(None, description="Number of retweets")
    reply_count: Optional[int] = Field(None, description="Number of replies")
    quote_count: Optional[int] = Field(None, description="Number of quotes")
    in_reply_to_user_id: Optional[str] = Field(None, description="User ID being replied to")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID")


class AppOutput(BaseAppOutput):
    """Output schema for search results."""
    posts: List[Post] = Field(description="List of matching posts")
    result_count: int = Field(description="Number of posts returned")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        """Initialize the X.com client with OAuth 2.0 access token."""
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError(
                "X_ACCESS_TOKEN not found. "
                "Please ensure the X.com integration is connected in Settings."
            )
        self.client = Client(access_token=access_token)
        print("X.com client initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Search recent posts on X.com."""
        print(f"Searching: {input_data.query[:100]}")

        try:
            posts = []
            users_map = {}

            for page in self.client.posts.search_recent(
                query=input_data.query,
                max_results=min(input_data.max_results, 100),
                sort_order=input_data.sort_order,
                tweet_fields=[
                    "id", "text", "author_id", "created_at",
                    "public_metrics", "in_reply_to_user_id", "conversation_id",
                    "article", "article_title",
                ],
                expansions=["author_id"],
                user_fields=["id", "name", "username"],
            ):
                # Build user lookup from expansions
                includes = getattr(page, "includes", None)
                if includes:
                    page_users = getattr(includes, "users", None) or []
                    for u in page_users:
                        if hasattr(u, 'model_dump'):
                            ud = u.model_dump()
                        elif isinstance(u, dict):
                            ud = u
                        else:
                            ud = u.__dict__ if hasattr(u, '__dict__') else {}
                        uid = str(ud.get("id", ""))
                        if uid:
                            users_map[uid] = {
                                "name": ud.get("name"),
                                "username": ud.get("username"),
                            }

                data = getattr(page, "data", None) or []
                for tweet in data:
                    # xdk returns pydantic models — try model_dump, then dict access, then getattr
                    if hasattr(tweet, 'model_dump'):
                        td = tweet.model_dump()
                    elif isinstance(tweet, dict):
                        td = tweet
                    else:
                        td = tweet.__dict__ if hasattr(tweet, '__dict__') else {}

                    metrics = td.get("public_metrics", {}) or {}
                    author_id = str(td.get("author_id", ""))
                    user_info = users_map.get(author_id, {})

                    posts.append(Post(
                        id=str(td.get("id", "")),
                        text=td.get("text", ""),
                        author_id=author_id,
                        author_name=user_info.get("name"),
                        author_username=user_info.get("username"),
                        created_at=td.get("created_at"),
                        like_count=metrics.get("like_count"),
                        retweet_count=metrics.get("retweet_count"),
                        reply_count=metrics.get("reply_count"),
                        quote_count=metrics.get("quote_count"),
                        in_reply_to_user_id=td.get("in_reply_to_user_id"),
                        conversation_id=td.get("conversation_id"),
                    ))

                    if len(posts) >= input_data.max_results:
                        break

                if len(posts) >= input_data.max_results:
                    break

            print(f"Found {len(posts)} posts")

            return AppOutput(
                posts=posts,
                result_count=len(posts),
            )

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        """Cleanup resources."""
        self.client = None
