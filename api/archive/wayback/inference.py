from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional
import requests
import logging

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    url: str = Field(
        description="the url to check for archived snapshots"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="specific timestamp to look up (format: YYYYMMDDhhmmss)"
    )


class AppOutput(BaseAppOutput):
    available: bool = Field(description="whether the url is archived")
    snapshot_url: Optional[str] = Field(default=None, description="url of the archived snapshot")
    snapshot_timestamp: Optional[str] = Field(default=None, description="timestamp of the snapshot")
    raw: dict = Field(description="full api response")


class App(BaseApp):
    async def setup(self, metadata):
        logging.info("archive wayback app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        params = {"url": input_data.url}

        if input_data.timestamp:
            params["timestamp"] = input_data.timestamp

        logging.info(f"checking wayback availability for: {input_data.url}")

        try:
            response = requests.get(
                "https://archive.org/wayback/available",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            snapshots = data.get("archived_snapshots", {})
            closest = snapshots.get("closest", {})

            available = closest.get("available", False)
            snapshot_url = closest.get("url") if available else None
            snapshot_timestamp = closest.get("timestamp") if available else None

            logging.info(f"wayback result for {input_data.url}: available={available}")

            return AppOutput(
                available=available,
                snapshot_url=snapshot_url,
                snapshot_timestamp=snapshot_timestamp,
                raw=data,
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"wayback api request failed: {e}")
            raise ValueError(f"wayback availability check failed: {str(e)}")
