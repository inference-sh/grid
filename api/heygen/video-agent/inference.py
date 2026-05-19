"""
HeyGen Video Agent - Prompt-to-video generation.

Generate complete videos from natural language prompts.
HeyGen's AI agent handles avatar selection, scripting, and editing.
"""

import logging
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .heygen_helper import (
    get_client,
    post_endpoint,
    poll_video,
    download_file,
)


OrientationType = Literal["landscape", "portrait"]


class AppInput(BaseAppInput):
    """Input for video agent generation."""

    prompt: str = Field(
        description="Natural language prompt describing the video to create (1-10,000 characters).",
        examples=[
            "Create a 30-second product demo for a new fitness app, with an energetic presenter",
            "Make an explainer video about how solar panels work, using a professional tone",
        ],
    )
    avatar_id: Optional[str] = Field(
        default=None,
        description="Specific avatar ID to use. If not set, the agent picks one.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Specific voice ID to use for narration.",
    )
    style_id: Optional[str] = Field(
        default=None,
        description="Visual template/style ID from HeyGen's style library.",
    )
    orientation: Optional[OrientationType] = Field(
        default=None,
        description="Video orientation. Auto-detected from prompt if not set.",
    )


class AppOutput(BaseAppOutput):
    """Output from video agent generation."""

    video: File = Field(description="The generated video.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Video agent prompt: {input_data.prompt[:100]}...")

        payload = {
            "prompt": input_data.prompt,
        }

        if input_data.avatar_id:
            payload["avatar_id"] = input_data.avatar_id
        if input_data.voice_id:
            payload["voice_id"] = input_data.voice_id
        if input_data.style_id:
            payload["style_id"] = input_data.style_id
        if input_data.orientation:
            payload["orientation"] = input_data.orientation

        async with get_client(timeout=300) as client:
            result = await post_endpoint(client, "/v3/video-agents", payload)

            video_id = result.get("video_id")
            session_id = result.get("session_id")
            self.logger.info(f"Session: {session_id}, video: {video_id}")

            if not video_id:
                raise RuntimeError(
                    f"Video agent session {session_id} did not return a video_id. "
                    f"Status: {result.get('status')}"
                )

            completed = await poll_video(client, video_id)

        video_url = completed["video_url"]
        video_path = await download_file(video_url)

        duration = completed.get("duration", 0)
        is_portrait = input_data.orientation == "portrait"
        width, height = (1080, 1920) if is_portrait else (1920, 1080)

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width,
                    height=height,
                    seconds=float(duration),
                    fps=24,
                )
            ]
        )

        self.logger.info(f"Video agent complete: {duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)
