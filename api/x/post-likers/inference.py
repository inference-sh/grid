import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional
from .x_helper import raise_api_error


class UserInfo(BaseAppInput):
    """A user who liked the post."""
    id: str = Field(description="User ID")
    name: str = Field(description="Display name")
    username: str = Field(description="Username")
    profile_image_url: Optional[str] = Field(None, description="Profile image URL")
    followers_count: Optional[int] = Field(None, description="Number of followers")


class AppInput(BaseAppInput):
    """Input schema for listing users who liked a post."""
    tweet_id: str = Field(description="The ID of the post to get likers for")
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of users to return"
    )


class AppOutput(BaseAppOutput):
    """Output schema for post likers."""
    tweet_id: str = Field(description="The post ID queried")
    users: List[UserInfo] = Field(description="Users who liked the post")
    count: int = Field(description="Number of users returned")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            users = []

            for page in self.client.posts.get_liking_users(
                id=input_data.tweet_id,
                max_results=input_data.max_results,
                user_fields=["id", "name", "username", "profile_image_url", "public_metrics"],
            ):
                data = getattr(page, "data", None) or []
                for user in data:
                    ud = user.model_dump() if hasattr(user, "model_dump") else user
                    metrics = ud.get("public_metrics", {}) or {}

                    users.append(UserInfo(
                        id=str(ud.get("id", "")),
                        name=ud.get("name", ""),
                        username=ud.get("username", ""),
                        profile_image_url=ud.get("profile_image_url"),
                        followers_count=metrics.get("followers_count"),
                    ))

                break  # first page only

            return AppOutput(
                tweet_id=input_data.tweet_id,
                users=users,
                count=len(users),
            )

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
