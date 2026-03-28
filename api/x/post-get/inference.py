import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List


class AppInput(BaseAppInput):
    tweet_id: str = Field(description="The ID of the tweet to retrieve")
    include_author: bool = Field(default=False, description="Include author profile info (name, avatar, verified)")
    include_media: bool = Field(default=False, description="Include media attachment URLs")


class AuthorInfo(BaseAppOutput):
    id: str = Field(description="Author user ID")
    name: str = Field(default="", description="Display name")
    username: str = Field(default="", description="Handle without @")
    profile_image_url: Optional[str] = Field(None, description="Avatar URL")
    verified: Optional[bool] = Field(None, description="Verified status")


class AppOutput(BaseAppOutput):
    id: str = Field(description="Post ID")
    text: str = Field(description="Post text content")
    author_id: str = Field(description="Author's user ID")
    author: Optional[AuthorInfo] = Field(None, description="Author profile (when include_author=true)")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    like_count: Optional[int] = Field(None, description="Number of likes")
    retweet_count: Optional[int] = Field(None, description="Number of retweets")
    reply_count: Optional[int] = Field(None, description="Number of replies")
    quote_count: Optional[int] = Field(None, description="Number of quotes")
    media_urls: Optional[List[str]] = Field(None, description="Media attachment URLs (when include_media=true)")
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
            tweet_fields = ["created_at", "public_metrics", "author_id"]
            if input_data.include_media:
                tweet_fields.append("attachments")

            expansions = []
            user_fields = []
            media_fields = []

            if input_data.include_author:
                expansions.append("author_id")
                user_fields = ["name", "username", "profile_image_url", "verified"]

            if input_data.include_media:
                expansions.append("attachments.media_keys")
                media_fields = ["url", "preview_image_url", "type"]

            kwargs = {"id": input_data.tweet_id, "tweet_fields": tweet_fields}
            if expansions:
                kwargs["expansions"] = expansions
            if user_fields:
                kwargs["user_fields"] = user_fields
            if media_fields:
                kwargs["media_fields"] = media_fields

            response = self.client.posts.get_by_id(**kwargs)

            data = response.data
            metrics = data.get("public_metrics", {})
            includes = getattr(response, "includes", {}) or {}

            author = None
            if input_data.include_author and includes.get("users"):
                u = includes["users"][0]
                author = AuthorInfo(
                    id=u["id"],
                    name=u.get("name", ""),
                    username=u.get("username", ""),
                    profile_image_url=u.get("profile_image_url"),
                    verified=u.get("verified"),
                )

            media_urls = None
            if input_data.include_media and includes.get("media"):
                media_urls = [
                    m.get("url") or m.get("preview_image_url", "")
                    for m in includes["media"]
                    if m.get("url") or m.get("preview_image_url")
                ]

            return AppOutput(
                id=data["id"],
                text=data.get("text", ""),
                author_id=data.get("author_id", ""),
                author=author,
                created_at=data.get("created_at"),
                like_count=metrics.get("like_count"),
                retweet_count=metrics.get("retweet_count"),
                reply_count=metrics.get("reply_count"),
                quote_count=metrics.get("quote_count"),
                media_urls=media_urls,
                post_url=f"https://x.com/i/web/status/{data['id']}"
            )
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
