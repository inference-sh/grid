import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
from .x_helper import raise_api_error


class DMEvent(BaseAppInput):
    """A single DM event."""
    id: str = Field(description="DM event ID")
    text: Optional[str] = Field(None, description="Message text")
    created_at: Optional[str] = Field(None, description="Event timestamp")
    sender_id: Optional[str] = Field(None, description="Sender user ID")
    event_type: Optional[str] = Field(None, description="Event type (e.g. MessageCreate)")


class AppInput(BaseAppInput):
    """Input schema for listing DM events."""
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum number of results (1-100)")


class AppOutput(BaseAppOutput):
    """Output schema for DM events."""
    events: List[DMEvent] = Field(description="List of DM events")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            print(f"Fetching DM events (max {input_data.max_results})")

            events = []

            for page in self.client.direct_messages.get_events(
                max_results=input_data.max_results,
                dm_event_fields=["id", "text", "created_at", "sender_id", "event_type"],
            ):
                data = getattr(page, "data", None) or []
                for event in data:
                    ed = event.model_dump() if hasattr(event, 'model_dump') else (event if isinstance(event, dict) else {})

                    events.append(DMEvent(
                        id=str(ed.get("id", "")),
                        text=ed.get("text"),
                        created_at=ed.get("created_at"),
                        sender_id=str(ed.get("sender_id", "")) if ed.get("sender_id") else None,
                        event_type=ed.get("event_type"),
                    ))

                    if len(events) >= input_data.max_results:
                        break

                break  # first page only

            print(f"Found {len(events)} DM events")

            return AppOutput(events=events)

        except Exception as e:
            raise_api_error(e)

    async def unload(self):
        self.client = None
