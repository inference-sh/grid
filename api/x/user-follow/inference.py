import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field


class AppInput(BaseAppInput):
    user_id: str = Field(description="The ID of the user to follow")


class AppOutput(BaseAppOutput):
    following: bool = Field(description="Whether now following the user")
    user_id: str = Field(description="ID of the user")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.users.follow(target_user_id=input_data.user_id)
            following = response.data.get("following", True)

            return AppOutput(following=following, user_id=input_data.user_id)
        except Exception as e:
            error_msg = str(e).lower()
            if "already following" in error_msg:
                return AppOutput(following=True, user_id=input_data.user_id)
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
