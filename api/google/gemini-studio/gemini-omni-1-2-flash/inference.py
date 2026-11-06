"""
Gemini Omni 1.2 Flash (gemini-omni-1.2-flash) via the Gemini API.

Video generation with native audio through the Interactions API (REST):
text-to-video, image-to-video, first and last frame interpolation, and
reference-to-video. One run; the request is built from which inputs are present.

Not exposed: video inputs (reference clips, editing, extension). Google refuses
uploaded video from the region the platform calls from ("Using uploaded video
or audio for video generation isn't available in your region").

Docs: https://ai.google.dev/gemini-api/docs/omni
"""

import asyncio
import base64
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, VideoMeta, function
from inferencesh.images import ImageLimits, fit_image
from pydantic import Field, model_validator

MODEL_ID = "gemini-omni-1.2-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Upstream documents JPEG and PNG image blocks and no size limits. These bounds
# keep a 4K frame untouched and the request body small.
IMAGE_LIMITS = ImageLimits(
    max_long_edge=4096,
    max_bytes=4 * 1024 * 1024,
    mime_types=("image/png", "image/jpeg"),
)

# Landscape (width, height) per resolution; swapped for 9:16.
DIMENSIONS = {
    "360p": (640, 360),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
# Video output tokens Google bills per second (pricing page footnote).
TOKENS_PER_SECOND = {"360p": 1931, "720p": 5792, "1080p": 8688, "4k": 17376}
FPS = 24

CREATE_ATTEMPTS = 3
FILE_POLL_SECONDS = 5
FILE_POLL_TIMEOUT_SECONDS = 600


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="What the video shows: scene, camera movement, lighting, mood and audio. "
        "Put exclusions in the prompt too (for example 'no dialogue')."
    )
    image: Optional[File] = Field(
        None,
        description="Image to animate. It becomes the starting frame of the video.",
    )
    last_frame: Optional[File] = Field(
        None,
        description="Ending frame. Needs image as the starting frame; the video transitions between the two. "
        "Durations up to 10 seconds.",
    )
    reference_images: List[File] = Field(
        default_factory=list,
        max_length=6,
        description="Up to 6 reference images that keep characters, objects or style consistent "
        "(up to 3 images per character for up to 2 characters).",
    )
    aspect_ratio: Literal["16:9", "9:16"] = Field(
        default="16:9",
        description="16:9 for landscape, 9:16 for portrait.",
    )
    resolution: Literal["360p", "720p", "1080p", "4k"] = Field(
        default="720p",
        description="Output resolution. 1080p and 4k are upscaled.",
    )
    duration: int = Field(
        default=10,
        ge=3,
        le=15,
        description="Video length in seconds, 3 to 15.",
    )

    @model_validator(mode="after")
    def _check_combinations(self):
        if self.last_frame is not None and self.image is None:
            raise ValueError("last_frame needs image as the starting frame")
        if self.last_frame is not None and self.duration > 10:
            raise ValueError("first and last frame videos support durations of 3 to 10 seconds")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video (MP4 with audio)")


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required.")
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(connect=30.0, read=900.0, write=300.0, pool=30.0),
            follow_redirects=True,
        )
        self.logger.info("Gemini Omni 1.2 Flash initialized")

    # ---- request ---------------------------------------------------------

    def _image_block(self, file: File) -> Dict[str, Any]:
        if not file.exists():
            raise RuntimeError(f"Input image does not exist: {file.path}")
        data, mime = fit_image(file.path, IMAGE_LIMITS)
        return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mime_type": mime}

    def _determine_mode(self, input_data: AppInput) -> str:
        has_refs = bool(input_data.reference_images)
        if input_data.image is not None and input_data.last_frame is not None:
            return "first_last_frame_with_references" if has_refs else "first_last_frame"
        if input_data.image is not None:
            return "image_to_video_with_references" if has_refs else "image_to_video"
        if has_refs:
            return "reference_to_video"
        return "text_to_video"

    def _build_request(self, input_data: AppInput) -> Dict[str, Any]:
        mode = self._determine_mode(input_data)
        content: List[Dict[str, Any]] = []
        # Upstream infers each media's role from the prompt. With one kind of
        # media the documented `task` settles it. When kinds are mixed no task
        # covers the combination, so each media gets a plain-language label.
        labelled = mode not in ("text_to_video", "image_to_video", "reference_to_video")

        def add(label: str, block: Dict[str, Any]) -> None:
            if labelled:
                content.append({"type": "text", "text": label})
            content.append(block)

        if input_data.image is not None:
            add("This image is the exact first frame of the video. The video starts on it:", self._image_block(input_data.image))
        if input_data.last_frame is not None:
            add("This image is the exact last frame of the video. The video ends on it:", self._image_block(input_data.last_frame))
        for i, ref in enumerate(input_data.reference_images, 1):
            add(
                "Reference image %d. Use it only for the appearance of a subject, object or style. "
                "It is not a frame of the video:" % i,
                self._image_block(ref),
            )
        content.append({"type": "text", "text": input_data.prompt})

        body: Dict[str, Any] = {
            "model": MODEL_ID,
            "input": content if len(content) > 1 else input_data.prompt,
            "response_format": {
                "type": "video",
                "aspect_ratio": input_data.aspect_ratio,
                "resolution": input_data.resolution,
                "duration": "%ds" % input_data.duration,
                # Inline delivery is limited to about 4 MB; a URI works at every size.
                # URI delivery needs the interaction stored (the default); store=false is refused.
                "delivery": "uri",
            },
        }
        task = {"image_to_video": "image_to_video", "reference_to_video": "reference_to_video"}.get(mode)
        if task:
            body["generation_config"] = {"video_config": {"task": task}}
        return body

    # ---- upstream calls --------------------------------------------------

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            err = response.json().get("error") or {}
            if isinstance(err, dict) and err.get("message"):
                return "%s %s: %s" % (response.status_code, err.get("status", ""), err["message"])
        except Exception:
            pass
        return "%s: %s" % (response.status_code, response.text[:500])

    @staticmethod
    def _retry_after(response: httpx.Response, fallback: float) -> float:
        try:
            return min(float(response.headers.get("retry-after", "")), 120.0)
        except ValueError:
            return fallback

    async def _create_interaction(self, body: Dict[str, Any]) -> Dict[str, Any]:
        # The create has no idempotency key: only a 429 (nothing was started) is retried.
        for attempt in range(1, CREATE_ATTEMPTS + 1):
            response = await self.client.post("/interactions", json=body)
            if response.status_code == 429 and attempt < CREATE_ATTEMPTS:
                delay = self._retry_after(response, 15.0 * attempt)
                self.logger.warning("Gemini API 429 on create (attempt %d), retrying in %.0fs", attempt, delay)
                await asyncio.sleep(delay)
                continue
            if response.status_code >= 400:
                raise RuntimeError("Gemini API error %s" % self._error_text(response))
            return response.json()
        raise RuntimeError("Gemini API rate limit: no capacity after %d attempts" % CREATE_ATTEMPTS)

    async def _get(self, url: str, **kwargs) -> httpx.Response:
        last = "no response"
        for attempt in range(1, 6):
            try:
                response = await self.client.get(url, **kwargs)
            except httpx.TransportError as exc:
                last = repr(exc)
            else:
                if response.status_code != 429 and response.status_code < 500:
                    return response
                last = self._error_text(response)
            await asyncio.sleep(min(2.0 * attempt, 10.0))
        raise RuntimeError("Gemini API error %s" % last)

    @staticmethod
    def _find_video(interaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates: List[Any] = [interaction.get("output_video")]
        for step in interaction.get("steps") or []:
            if isinstance(step, dict) and step.get("type") == "model_output":
                candidates.extend(step.get("content") or [])
        for block in candidates:
            if isinstance(block, dict) and (block.get("uri") or block.get("data")):
                if block.get("type") in (None, "video"):
                    return block
        return None

    @staticmethod
    def _summary(interaction: Dict[str, Any]) -> Dict[str, Any]:
        """The response without media payloads, for logs and error messages."""
        steps = []
        for step in interaction.get("steps") or []:
            if not isinstance(step, dict):
                continue
            blocks = []
            for block in step.get("content") or []:
                if isinstance(block, dict):
                    blocks.append({k: (v if k != "data" else "<%d chars>" % len(v or "")) for k, v in block.items()})
            steps.append({"type": step.get("type"), "content": blocks})
        out = {k: v for k, v in interaction.items() if k not in ("steps", "output_video")}
        out["steps"] = [s for s in steps if s["type"] != "user_input"]
        return out

    async def _save_video(self, block: Dict[str, Any]) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        if block.get("data"):
            with open(path, "wb") as f:
                f.write(base64.b64decode(block["data"]))
            return path

        match = re.search(r"files/([^/:?]+)", block["uri"])
        if not match:
            raise RuntimeError("Unexpected video URI from Gemini API: %s" % block["uri"][:200])
        file_id = match.group(1)

        waited = 0
        while True:
            response = await self._get("/files/%s" % file_id)
            if response.status_code >= 400:
                raise RuntimeError("Gemini API error reading the video file %s" % self._error_text(response))
            state = response.json().get("state")
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini API failed to process the generated video")
            if waited >= FILE_POLL_TIMEOUT_SECONDS:
                raise RuntimeError("Generated video was not ready after %ds (state %s)" % (waited, state))
            await asyncio.sleep(FILE_POLL_SECONDS)
            waited += FILE_POLL_SECONDS

        response = await self._get("/files/%s:download" % file_id, params={"alt": "media"})
        if response.status_code >= 400:
            raise RuntimeError("Gemini API error downloading the video %s" % self._error_text(response))
        with open(path, "wb") as f:
            f.write(response.content)
        return path

    # ---- usage -----------------------------------------------------------

    @staticmethod
    def _usage_tokens(usage: Dict[str, Any], resolution: str, seconds: int) -> Tuple[int, int, int]:
        """(input tokens, video output tokens, text output tokens incl. thinking)."""
        by_modality = {
            str(item.get("modality", "")).lower(): int(item.get("tokens") or 0)
            for item in usage.get("output_tokens_by_modality") or []
            if isinstance(item, dict)
        }
        input_tokens = int(usage.get("total_input_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        thought_tokens = int(usage.get("total_thought_tokens") or 0)
        if "video" in by_modality:
            video_tokens = by_modality["video"]
        else:
            video_tokens = TOKENS_PER_SECOND[resolution] * seconds
        text_tokens = max(output_tokens - video_tokens, 0) + thought_tokens
        return input_tokens, video_tokens, text_tokens

    # ---- run -------------------------------------------------------------

    @function(annotations={"title": "Generate video", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate a video with audio from a text prompt, an image, first and last frames, or reference media."""
        mode = self._determine_mode(input_data)
        self.logger.info(
            "mode=%s aspect_ratio=%s resolution=%s duration=%ds reference_images=%d prompt=%r",
            mode,
            input_data.aspect_ratio,
            input_data.resolution,
            input_data.duration,
            len(input_data.reference_images),
            input_data.prompt[:100],
        )
        body = self._build_request(input_data)

        interaction = await self._create_interaction(body)
        usage = interaction.get("usage") or {}
        self.logger.info(
            "interaction id=%s status=%s usage=%s", interaction.get("id"), interaction.get("status"), usage
        )

        block = self._find_video(interaction)
        if block is None:
            self.logger.error("no video in response: %s", self._summary(interaction))
            raise RuntimeError(
                "No video was generated (status %s). The prompt or an input may have been blocked by "
                "Google's safety filters." % interaction.get("status")
            )

        video_path = await self._save_video(block)
        self.logger.info("video saved: %d bytes", os.path.getsize(video_path))

        width, height = DIMENSIONS[input_data.resolution]
        if input_data.aspect_ratio == "9:16":
            width, height = height, width
        input_tokens, video_tokens, text_tokens = self._usage_tokens(usage, input_data.resolution, input_data.duration)

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=[TextMeta(tokens=input_tokens)],
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=input_data.resolution,
                        # Google bills the requested length, not the delivered file's.
                        seconds=float(input_data.duration),
                        fps=FPS,
                        extra={"tokens": video_tokens, "mode": mode},
                    ),
                    TextMeta(tokens=text_tokens),
                ],
            ),
        )

    async def unload(self):
        await self.client.aclose()

    async def on_cancel(self):
        return True
