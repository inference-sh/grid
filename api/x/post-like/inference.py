import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field


class AppInput(BaseAppInput):
    post_id: str = Field(description="The ID of the post to like")


class AppOutput(BaseAppOutput):
    liked: bool = Field(description="Whether the post was successfully liked")
    post_id: str = Field(description="ID of the liked post")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.likes.create(tweet_id=input_data.post_id)
            liked = response.data.get("liked", True)

            return AppOutput(liked=liked, post_id=input_data.post_id)
        except Exception as e:
            error_msg = str(e).lower()
            if "already liked" in error_msg:
                return AppOutput(liked=True, post_id=input_data.post_id)
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
