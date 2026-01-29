import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field


class AppInput(BaseAppInput):
    post_id: str = Field(description="The ID of the post to delete")


class AppOutput(BaseAppOutput):
    deleted: bool = Field(description="Whether the post was successfully deleted")
    post_id: str = Field(description="ID of the deleted post")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.posts.delete(id=input_data.post_id)
            deleted = response.data.get("deleted", True)

            return AppOutput(deleted=deleted, post_id=input_data.post_id)
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
