"""
HeyGen Avatar Video - Generate talking avatar videos.

Create videos with HeyGen's digital avatars speaking from a script,
with configurable voice, resolution, and aspect ratio.
"""

import logging
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from pydantic import BaseModel
from typing import List

from .heygen_helper import (
    get_client,
    post_endpoint,
    poll_video,
    download_file,
    list_avatars,
    list_voices,
    get_look,
)


EngineType = Literal["avatar_iv", "avatar_v"]
ExpressivenessType = Literal["high", "medium", "low"]
ResolutionType = Literal["720p", "1080p", "4k"]
AspectRatioType = Literal["16:9", "9:16"]
OutputFormatType = Literal["mp4", "webm"]


class AppInput(BaseAppInput):
    """Input for avatar video generation."""

    avatar_id: str = Field(
        description="HeyGen avatar ID. Use the HeyGen dashboard or API to find available avatars."
    )
    script: str = Field(
        description="Text for the avatar to speak.",
        examples=["Hello! Welcome to our product demo. Let me show you how it works."],
    )
    engine: EngineType = Field(
        default="avatar_iv",
        description="Rendering engine. Avatar V offers more natural motion and lip-sync but requires eligible avatar looks. Avatar IV is the default.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for the avatar. If not set, the avatar's default voice is used.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Title for the video.",
    )
    resolution: ResolutionType = Field(
        default="1080p",
        description="Video resolution.",
    )
    aspect_ratio: AspectRatioType = Field(
        default="16:9",
        description="Video aspect ratio.",
    )
    expressiveness: Optional[ExpressivenessType] = Field(
        default=None,
        description="Avatar expressiveness level. Only applies to photo avatars.",
    )
    motion_prompt: Optional[str] = Field(
        default=None,
        description="Natural language description of avatar motion. Only applies to photo avatars.",
    )
    remove_background: Optional[bool] = Field(
        default=None,
        description="Remove the video background.",
    )
    output_format: OutputFormatType = Field(
        default="mp4",
        description="Output video format.",
    )


class AppOutput(BaseAppOutput):
    """Output from avatar video generation."""

    video: File = Field(description="The generated avatar video.")


RESOLUTION_DIMENSIONS = {
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280)},
    "1080p": {"16:9": (1920, 1080), "9:16": (1080, 1920)},
    "4k": {"16:9": (3840, 2160), "9:16": (2160, 3840)},
}


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Creating avatar video ({input_data.engine}): {input_data.script[:100]}...")

        async with get_client() as client:
            # Fetch look metadata for eligibility check and pricing
            look = await get_look(client, input_data.avatar_id)
            avatar_type = look.get("avatar_type", "unknown")
            self.logger.info(f"Avatar type: {avatar_type}")

            # Avatar V eligibility check
            if input_data.engine == "avatar_v":
                engines = look.get("supported_api_engines", [])
                if "avatar_v" not in engines:
                    raise RuntimeError(
                        f"Avatar look {input_data.avatar_id} does not support Avatar V. "
                        f"Supported engines: {engines}"
                    )
                self.logger.info(f"Avatar V eligibility confirmed for {input_data.avatar_id}")

            payload = {
                "type": "avatar",
                "avatar_id": input_data.avatar_id,
                "script": input_data.script,
                "aspect_ratio": input_data.aspect_ratio,
                "resolution": input_data.resolution,
                "output_format": input_data.output_format,
            }

            if input_data.engine == "avatar_v":
                payload["engine"] = {"type": "avatar_v"}
                if input_data.motion_prompt:
                    self.logger.warning("motion_prompt is not supported with Avatar V, ignoring")
                if input_data.expressiveness:
                    self.logger.warning("expressiveness is not supported with Avatar V, ignoring")
            else:
                if input_data.expressiveness:
                    payload["expressiveness"] = input_data.expressiveness
                if input_data.motion_prompt:
                    payload["motion_prompt"] = input_data.motion_prompt

            if input_data.voice_id:
                payload["voice_id"] = input_data.voice_id
            if input_data.title:
                payload["title"] = input_data.title
            if input_data.remove_background is not None:
                payload["remove_background"] = input_data.remove_background

            result = await post_endpoint(client, "/v3/videos", payload)
            video_id = result["video_id"]
            self.logger.info(f"Video created: {video_id}, polling...")

            completed = await poll_video(client, video_id)

        video_url = completed["video_url"]
        suffix = ".webm" if input_data.output_format == "webm" else ".mp4"
        video_path = await download_file(video_url, suffix=suffix)

        duration = completed.get("duration", 0)
        dims = RESOLUTION_DIMENSIONS.get(input_data.resolution, {}).get(
            input_data.aspect_ratio, (1920, 1080)
        )

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=dims[0],
                    height=dims[1],
                    seconds=float(duration),
                    fps=24,
                    extra={"avatar_type": avatar_type},
                )
            ]
        )

        self.logger.info(f"Avatar video complete: {duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def list_resources(self, input_data: "ListResourcesInput") -> "ListResourcesOutput":
        """List available HeyGen avatars and voices."""
        avatars_out = []
        voices_out = []

        async with get_client() as client:
            if input_data.resource_type in ("avatars", "both"):
                raw = await list_avatars(client, input_data.limit)
                for a in raw:
                    engines = a.get("supported_api_engines", [])
                    avatars_out.append(ResourceItem(
                        id=a.get("avatar_look_id", a.get("id", "")),
                        name=a.get("name", ""),
                        avatar_type=a.get("avatar_type", None),
                        supported_engines=engines if engines else None,
                    ))

            if input_data.resource_type in ("voices", "both"):
                raw = await list_voices(client, input_data.limit, engine="starfish")
                for v in raw:
                    voices_out.append(ResourceItem(
                        id=v.get("voice_id", ""),
                        name=v.get("name", ""),
                        avatar_type=v.get("language", None),
                    ))

        return ListResourcesOutput(avatars=avatars_out, voices=voices_out)


class ListResourcesInput(BaseAppInput):
    """List available avatars and voices."""
    resource_type: Literal["avatars", "voices", "both"] = Field(
        default="both", description="Which resources to list."
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results per type.")


class ResourceItem(BaseModel):
    id: str = Field(description="Resource ID")
    name: str = Field(description="Resource name")
    avatar_type: Optional[str] = Field(default=None, description="Avatar type or voice language")
    supported_engines: Optional[List[str]] = Field(default=None, description="Supported rendering engines (e.g. avatar_iv, avatar_v)")


class ListResourcesOutput(BaseAppOutput):
    avatars: List[ResourceItem] = Field(default_factory=list, description="Available avatars")
    voices: List[ResourceItem] = Field(default_factory=list, description="Available voices")
