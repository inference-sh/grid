"""
HeyGen Create Avatar - Create avatars from video, photo, or text prompt.

Supports three creation methods:
- Digital Twin: from video footage of a person
- Photo Avatar: from a single portrait image
- Prompt: AI-generated from a text description
"""

import logging
from typing import Optional, Literal, List

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta
from pydantic import Field, BaseModel

from .heygen_helper import get_client, post_endpoint, build_asset_ref, poll_avatar


AvatarType = Literal["digital_twin", "photo", "prompt"]


class AppInput(BaseAppInput):
    """Create a new HeyGen avatar."""

    type: AvatarType = Field(
        description="Avatar creation method. 'digital_twin' from video, 'photo' from image, 'prompt' from text description."
    )
    name: str = Field(
        description="Display name for the avatar."
    )
    file: Optional[File] = Field(
        default=None,
        description="Video file (digital_twin) or image file (photo). Required for digital_twin and photo types.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Text description of the avatar to generate. Required for prompt type.",
        examples=["A professional woman in her 30s with short dark hair, wearing a navy blazer, in a modern office setting"],
    )
    avatar_group_id: Optional[str] = Field(
        default=None,
        description="Attach this look to an existing avatar group/character identity.",
    )


class AvatarItem(BaseModel):
    id: str = Field(description="Look ID — use this as avatar_id in avatar-video")
    name: str = Field(description="Display name")
    avatar_type: str = Field(description="Avatar type (digital_twin, photo_avatar, studio_avatar)")
    group_id: str = Field(description="Avatar group/character identity ID")
    preview_image_url: Optional[str] = Field(default=None, description="Preview image URL")
    supported_engines: Optional[List[str]] = Field(default=None, description="Supported rendering engines")


class AvatarGroup(BaseModel):
    id: str = Field(description="Group ID")
    name: str = Field(description="Group name")
    consent_status: Optional[str] = Field(default=None, description="Consent status (digital twins only)")
    consent_url: Optional[str] = Field(default=None, description="Consent page URL if consent is pending (digital twins only)")


class AppOutput(BaseAppOutput):
    """Created avatar details."""

    avatar: AvatarItem = Field(description="The created avatar look")
    group: AvatarGroup = Field(description="The avatar group/character identity")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Creating {input_data.type} avatar: {input_data.name}")

        payload: dict = {
            "type": input_data.type,
            "name": input_data.name,
        }

        if input_data.type in ("digital_twin", "photo"):
            if not input_data.file:
                raise RuntimeError(f"{input_data.type} requires a file (video or image)")
            payload["file"] = build_asset_ref(input_data.file)

        elif input_data.type == "prompt":
            if not input_data.prompt:
                raise RuntimeError("prompt type requires a prompt description")
            payload["prompt"] = input_data.prompt

        if input_data.avatar_group_id:
            payload["avatar_group_id"] = input_data.avatar_group_id

        async with get_client() as client:
            result = await post_endpoint(client, "/v3/avatars", payload)

            item = result.get("avatar_item", {})
            group = result.get("avatar_group", {})
            look_id = item.get("id", "")

            self.logger.info(f"Avatar submitted: look_id={look_id}, group_id={group.get('id')}")

            # Poll until the avatar is ready (prompt/photo avatars need processing)
            status = item.get("status", "")
            if status and status != "completed":
                self.logger.info(f"Avatar status: {status}, polling until ready...")
                item = await poll_avatar(client, look_id)
                self.logger.info(f"Avatar ready: {item.get('name')}")

            # Digital twins need consent — initiate the flow if not already approved
            consent_url = None
            if input_data.type == "digital_twin" and group.get("consent_status") != "approved":
                self.logger.info(f"Digital twin consent pending for group {group.get('id')}")
                try:
                    consent_data = await post_endpoint(
                        client, f"/v3/avatars/{group['id']}/consent", {}
                    )
                    consent_url = consent_data.get("url")
                    self.logger.info(f"Consent URL: {consent_url}")
                except Exception as e:
                    self.logger.warning(f"Could not initiate consent flow: {e}")

        engines = item.get("supported_api_engines", [])

        # Avatar creation is $1.00 flat per call
        output_meta = OutputMeta(outputs=[RawMeta(cost=100)])

        return AppOutput(
            avatar=AvatarItem(
                id=item.get("id", ""),
                name=item.get("name", input_data.name),
                avatar_type=item.get("avatar_type", input_data.type),
                group_id=item.get("group_id", group.get("id", "")),
                preview_image_url=item.get("preview_image_url"),
                supported_engines=engines if engines else None,
            ),
            group=AvatarGroup(
                id=group.get("id", ""),
                name=group.get("name", input_data.name),
                consent_status=group.get("consent_status"),
                consent_url=consent_url,
            ),
            output_meta=output_meta,
        )
