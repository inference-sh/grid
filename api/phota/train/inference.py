import logging
import time
from typing import List, Literal, Optional
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

from .phota_helper import get_api_key, get_headers, BASE_URL

import requests as http


class TrainInput(BaseAppInput):
    images: List[File] = Field(description="30-50 face images of the subject", min_length=30, max_length=50)
    wait: bool = Field(default=True, description="Wait for training to complete (polls status)")
    poll_interval: int = Field(default=10, ge=5, le=60, description="Seconds between status polls (if wait=True)")


class TrainOutput(BaseAppOutput):
    profile_id: str = Field(description="Unique profile identifier")
    status: str = Field(description="Final profile status (READY if wait=True)")


class StatusInput(BaseAppInput):
    profile_id: str = Field(description="Profile ID to check")


class StatusOutput(BaseAppOutput):
    profile_id: str = Field(description="Profile identifier")
    status: str = Field(description="Current status: VALIDATING, QUEUING, IN_PROGRESS, READY, ERROR, INACTIVE")
    message: Optional[str] = Field(None, description="Error message if status is ERROR")


class ListOutput(BaseAppOutput):
    profiles: List[dict] = Field(description="List of profiles with profile_id and status")


class ListInput(BaseAppInput):
    pass


class DeleteInput(BaseAppInput):
    profile_id: str = Field(description="Profile ID to delete")


class DeleteOutput(BaseAppOutput):
    profile_id: str = Field(description="Deleted profile identifier")
    status: str = Field(description="Result status")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Phota Train app initialized")

    async def run(self, input_data: TrainInput) -> TrainOutput:
        self.logger.info(f"Creating profile with {len(input_data.images)} images")

        # Phota API needs publicly accessible URLs — use the uploaded file URIs
        image_urls = []
        for img in input_data.images:
            if img.uri and img.uri.startswith("http"):
                image_urls.append(img.uri)
            else:
                raise RuntimeError(f"Image must be a URL, got local path: {img.path}")

        resp = http.post(
            f"{BASE_URL}/profiles/add",
            json={"image_urls": image_urls},
            headers=get_headers(),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        profile_id = data["profile_id"]

        self.logger.info(f"Profile created: {profile_id}")

        if not input_data.wait:
            return TrainOutput(profile_id=profile_id, status="SUBMITTED")

        # Poll until READY or ERROR
        while True:
            time.sleep(input_data.poll_interval)
            status_resp = http.get(
                f"{BASE_URL}/profiles/{profile_id}/status",
                headers=get_headers(),
                timeout=30,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data["status"]

            self.logger.info(f"Profile {profile_id}: {status}")

            if status == "READY":
                return TrainOutput(profile_id=profile_id, status="READY")
            elif status == "ERROR":
                msg = status_data.get("message", "Unknown error")
                raise RuntimeError(f"Training failed: {msg}")
            elif status == "INACTIVE":
                raise RuntimeError(f"Profile became INACTIVE during training")

    async def status(self, input_data: StatusInput) -> StatusOutput:
        self.logger.info(f"Checking status: {input_data.profile_id}")

        resp = http.get(
            f"{BASE_URL}/profiles/{input_data.profile_id}/status",
            headers=get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return StatusOutput(
            profile_id=data["profile_id"],
            status=data["status"],
            message=data.get("message"),
        )

    async def list_profiles(self, input_data: ListInput) -> ListOutput:
        self.logger.info("Listing all profiles")

        resp = http.get(
            f"{BASE_URL}/profiles/ids",
            headers=get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        self.logger.info(f"Found {len(data['profiles'])} profiles")

        return ListOutput(profiles=data["profiles"])

    async def delete(self, input_data: DeleteInput) -> DeleteOutput:
        self.logger.info(f"Deleting profile: {input_data.profile_id}")

        resp = http.delete(
            f"{BASE_URL}/profiles/{input_data.profile_id}",
            headers=get_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        self.logger.info(f"Deleted: {data['profile_id']}")

        return DeleteOutput(
            profile_id=data["profile_id"],
            status=data["status"],
        )
