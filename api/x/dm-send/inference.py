import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field


class AppInput(BaseAppInput):
    recipient_id: str = Field(description="User ID of the recipient")
    text: str = Field(description="Message text to send", min_length=1)


class AppOutput(BaseAppOutput):
    event_id: str = Field(description="ID of the DM event")
    recipient_id: str = Field(description="ID of the recipient")
    sent: bool = Field(description="Whether the message was sent successfully")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            response = self.client.direct_messages.create(
                participant_id=input_data.recipient_id,
                body={"text": input_data.text}
            )

            event_id = getattr(response.data, "dm_event_id", None) or getattr(response.data, "id", "")

            return AppOutput(
                event_id=event_id,
                recipient_id=input_data.recipient_id,
                sent=True
            )
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
