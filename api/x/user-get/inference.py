import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    user_id: Optional[str] = Field(None, description="User ID to look up")
    username: Optional[str] = Field(None, description="Username to look up (without @)")


class AppOutput(BaseAppOutput):
    id: str = Field(description="User ID")
    username: str = Field(description="Username")
    name: str = Field(description="Display name")
    description: Optional[str] = Field(None, description="User bio")
    profile_image_url: Optional[str] = Field(None, description="Profile image URL")
    followers_count: Optional[int] = Field(None, description="Number of followers")
    following_count: Optional[int] = Field(None, description="Number of accounts following")
    tweet_count: Optional[int] = Field(None, description="Number of tweets")
    verified: Optional[bool] = Field(None, description="Whether the user is verified")
    created_at: Optional[str] = Field(None, description="Account creation date")
    profile_url: str = Field(description="URL to user profile")


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
            user_fields = ["description", "profile_image_url", "public_metrics", "verified", "created_at"]

            if input_data.user_id:
                response = self.client.users.get(id=input_data.user_id, user_fields=user_fields)
            else:
                response = self.client.users.get_by_username(
                    username=input_data.username.lstrip("@"),
                    user_fields=user_fields
                )

            data = response.data
            metrics = data.get("public_metrics", {})

            return AppOutput(
                id=data["id"],
                username=data["username"],
                name=data.get("name", ""),
                description=data.get("description"),
                profile_image_url=data.get("profile_image_url"),
                followers_count=metrics.get("followers_count"),
                following_count=metrics.get("following_count"),
                tweet_count=metrics.get("tweet_count"),
                verified=data.get("verified"),
                created_at=data.get("created_at"),
                profile_url=f"https://x.com/{data['username']}"
            )
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
