import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from .x_helper import raise_api_error


class AppInput(BaseAppInput):
    """Input schema for removing a bookmark."""
    tweet_id: str = Field(description="The ID of the post to remove from bookmarks")


class AppOutput(BaseAppOutput):
    """Output schema for bookmark removal result."""
    removed: bool = Field(description="Whether the bookmark was successfully removed")


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

            self.client.users.delete_bookmark(
                id=user_id,
                tweet_id=input_data.tweet_id,
            )

            return AppOutput(removed=True)

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
