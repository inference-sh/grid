import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional
from .x_helper import raise_api_error


class UserInfo(BaseAppInput):
    """A follower profile."""
    id: str = Field(description="User ID")
    name: str = Field(description="Display name")
    username: str = Field(description="Username")
    profile_image_url: Optional[str] = Field(None, description="Profile image URL")
    description: Optional[str] = Field(None, description="User bio")
    followers_count: Optional[int] = Field(None, description="Number of followers")
    following_count: Optional[int] = Field(None, description="Number following")
    tweet_count: Optional[int] = Field(None, description="Number of tweets")


class AppInput(BaseAppInput):
    """Input schema for listing followers."""
    username: Optional[str] = Field(None, description="Username to get followers for (without @)")
    user_id: Optional[str] = Field(None, description="User ID to get followers for")
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of followers to return"
    )


class AppOutput(BaseAppOutput):
    """Output schema for followers list."""
    user_id: str = Field(description="The user ID whose followers were listed")
    users: List[UserInfo] = Field(description="Follower profiles")
    count: int = Field(description="Number of followers returned")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            user_id = input_data.user_id

            if not user_id and input_data.username:
                resp = self.client.users.get_by_username(
                    username=input_data.username.lstrip("@")
                )
                user_id = resp.data.id

            if not user_id:
                resp = self.client.users.get_me()
                user_id = resp.data.id

            user_id = str(user_id)
            users = []

            for page in self.client.users.get_followers(
                id=user_id,
                max_results=input_data.max_results,
                user_fields=[
                    "id", "name", "username", "profile_image_url",
                    "description", "public_metrics", "created_at",
                ],
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
                        description=ud.get("description"),
                        followers_count=metrics.get("followers_count"),
                        following_count=metrics.get("following_count"),
                        tweet_count=metrics.get("tweet_count"),
                    ))

                break  # first page only

            return AppOutput(
                user_id=user_id,
                users=users,
                count=len(users),
            )

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
