"""
Shared implementation for the Seedance 2.0 app family.

The six Seedance 2.0 apps (full / fast / mini, each in a plain and a "studio"
variant) differ only in four things: display name, model ID, the set of
resolutions the model supports, and whether references are routed through the
BytePlus private asset library. Everything else — mode detection, content
building, task polling, output probing, usage metadata — is identical, so it
lives here.

Each app keeps its own AppInput/AppOutput and a thin `run` override, because
those are what generate the app's public API schema and they differ per app
(resolution enum members, field descriptions).

Two classes:
    SeedanceApp        — passes reference URLs through directly, and always
                         uses the standard safety-filtered endpoint
    SeedanceStudioApp  — uploads every reference to the asset library first
                         and passes asset:// URIs instead; the only variant
                         that may expose safety_filter / a custom endpoint

Not used by the Seedance 1.x apps: those take a different input shape
(no references, no ratio/audio) and encode parameters into the prompt text
rather than as top-level request fields.
"""

import logging
from typing import Any, ClassVar, Optional

from inferencesh import (
    BaseApp,
    File,
    OutputMeta,
    VideoMeta,
    VideoResolution,
    ImageMeta,
    AudioMeta,
)

from .byteplus_helper import (
    setup_byteplus_client,
    create_content_task,
    poll_task_status,
    cancel_task,
    download_video,
    build_text_content,
    build_image_content,
    build_video_content,
    build_audio_content,
    probe_video,
)
from .asset_library_helper import (
    setup_asset_client,
    create_asset_group,
    upload_and_activate,
)


RESOLUTION_MAP = {
    '480p': VideoResolution.VIDEO_RES480_P,
    '720p': VideoResolution.VIDEO_RES720_P,
    '1080p': VideoResolution.VIDEO_RES1080_P,
    '4k': VideoResolution.VIDEO_RES4_K,
}


class SeedanceApp(BaseApp):
    """Seedance 2.0 video generation via the BytePlus ARK SDK.

    Subclasses set the class attributes below and define their own
    AppInput/AppOutput plus a `run` that delegates to `super().run(...)`.
    """

    # --- per-app knobs ---
    # ClassVar throughout: BaseApp is a Pydantic model, so a bare annotation
    # would turn these into request fields and leak into the public schema.
    display_name: ClassVar[str] = "Seedance 2.0"
    model_id: ClassVar[str] = ""
    # Marks output metadata so pricing/analytics can tell the variants apart.
    is_studio: ClassVar[bool] = False
    # The app's own AppOutput class. Set by each app so the base can build the
    # return value without importing app-specific schemas.
    OutputType: ClassVar[Any] = None

    async def setup(self, metadata):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = setup_byteplus_client()

        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info(f"{self.display_name} initialized with model: {self.model_id}")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)
        return True

    # --- content building ---

    def _determine_mode(self, input_data) -> str:
        """Determine the generation mode from input."""
        has_refs = input_data.reference_images or input_data.reference_videos or input_data.reference_audios

        if has_refs:
            return "multimodal-reference"
        elif input_data.image and input_data.end_image:
            return "first-last-frame"
        elif input_data.image:
            return "image-to-video"
        else:
            return "text-to-video"

    async def _resolve_uri(self, file: File, asset_type: str, label: str) -> str:
        """Resolve an input file to the URI handed to the generation API.

        The plain variant passes the file's own URL through. The studio variant
        overrides this to upload the file to the asset library first.
        """
        if not file or not file.exists():
            raise RuntimeError(f"{label} does not exist: {getattr(file, 'path', file)}")
        return file.uri

    async def _prepare_generation(self, input_data) -> None:
        """Hook run before content building. Studio uses it to open its group."""
        return None

    async def _build_content(self, input_data, mode: str) -> list:
        """Build the content list for the BytePlus API."""
        content = []

        if input_data.prompt:
            content.append(build_text_content(input_data.prompt))

        if mode == "first-last-frame":
            first_uri = await self._resolve_uri(input_data.image, "Image", "First-frame image")
            last_uri = await self._resolve_uri(input_data.end_image, "Image", "Last-frame image")
            content.append(build_image_content(first_uri, role="first_frame"))
            content.append(build_image_content(last_uri, role="last_frame"))

        elif mode == "image-to-video":
            first_uri = await self._resolve_uri(input_data.image, "Image", "Input image")
            content.append(build_image_content(first_uri, role="first_frame"))

        elif mode == "multimodal-reference":
            for ref_img in input_data.reference_images:
                if ref_img.exists():
                    uri = await self._resolve_uri(ref_img, "Image", "Reference image")
                    content.append(build_image_content(uri, role="reference_image"))

            for ref_vid in input_data.reference_videos:
                if ref_vid.exists():
                    uri = await self._resolve_uri(ref_vid, "Video", "Reference video")
                    content.append(build_video_content(uri))

            if input_data.reference_audios:
                has_visual = input_data.reference_images or input_data.reference_videos
                if not has_visual:
                    raise RuntimeError("Audio reference requires at least one image or video reference.")
                for ref_aud in input_data.reference_audios:
                    if ref_aud.exists():
                        uri = await self._resolve_uri(ref_aud, "Audio", "Reference audio")
                        content.append(build_audio_content(uri))

        return content

    # --- output metadata ---

    def _build_output_meta(self, input_data, result, mode: str, video_path: str) -> OutputMeta:
        """Build output metadata from the generation result."""
        # Probe actual output video for real dimensions, fps, frame count
        probe = probe_video(video_path)
        width = probe.get("width", 1280)
        height = probe.get("height", 720)
        fps = probe.get("fps", 24)
        actual_duration = probe.get("seconds", float(input_data.duration) if input_data.duration > 0 else 5.0)
        actual_resolution = getattr(result, 'resolution', input_data.resolution.value)
        actual_ratio = getattr(result, 'ratio', input_data.ratio.value)
        if actual_ratio == 'adaptive':
            actual_ratio = '16:9'
        seed = getattr(result, 'seed', None)

        usage = getattr(result, 'usage', None)
        completion_tokens = None
        total_tokens = None
        if usage:
            completion_tokens = getattr(usage, 'completion_tokens', None)
            total_tokens = getattr(usage, 'total_tokens', None)
        self.logger.info(f"BytePlus usage — completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, mode: {mode}, probe: {width}x{height}@{fps} {actual_duration:.3f}s")

        resolution_enum = RESOLUTION_MAP.get(actual_resolution, VideoResolution.VIDEO_RES720_P)

        # Build input metadata for pricing
        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.end_image:
            input_metas.append(ImageMeta())
        if input_data.reference_images:
            for _ in input_data.reference_images:
                input_metas.append(ImageMeta())
        if input_data.reference_videos:
            for ref_vid in input_data.reference_videos:
                if ref_vid and ref_vid.exists():
                    ref_probe = probe_video(ref_vid.path)
                    ref_frames = ref_probe.get("nb_frames", 0)
                    ref_fps = ref_probe.get("fps", 24)
                    ref_seconds = ref_frames / ref_fps if ref_fps > 0 else 0.0
                    input_metas.append(VideoMeta(seconds=ref_seconds))
                else:
                    input_metas.append(VideoMeta())
        if input_data.reference_audios:
            for _ in input_data.reference_audios:
                input_metas.append(AudioMeta())

        extra = {
            "mode": mode,
            "ratio": actual_ratio,
            "generate_audio": input_data.generate_audio,
            "seed": seed,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if self.is_studio:
            extra["studio"] = True

        return OutputMeta(
            inputs=input_metas,
            outputs=[
                VideoMeta(
                    width=width,
                    height=height,
                    resolution=resolution_enum,
                    seconds=float(actual_duration),
                    fps=fps,
                    extra=extra,
                )
            ]
        )

    # --- generation ---

    def _select_model(self, input_data) -> str:
        """Always the standard, safety-filtered endpoint.

        Plain apps do not expose safety_filter and have no custom endpoint;
        only the studio variants can opt out. See SeedanceStudioApp.
        """
        return self.model_id

    async def run(self, input_data, metadata):
        """Generate a video. Subclasses re-declare this with typed signatures."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            mode = self._determine_mode(input_data)
            suffix = " (studio)" if self.is_studio else ""
            self.logger.info(f"Starting {mode} generation{suffix}")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Ratio: {input_data.ratio.value}, Duration: {input_data.duration}s, Audio: {input_data.generate_audio}")

            await self._prepare_generation(input_data)

            content = await self._build_content(input_data, mode)

            api_params = {
                "resolution": input_data.resolution.value,
                "ratio": input_data.ratio.value,
                "duration": input_data.duration,
                "generate_audio": input_data.generate_audio,
                "seed": input_data.seed,
                "watermark": input_data.watermark,
            }
            if input_data.safety_identifier:
                api_params["safety_identifier"] = input_data.safety_identifier

            self.current_task_id = create_content_task(
                self.client,
                model=self._select_model(input_data),
                content=content,
                logger=self.logger,
                **api_params,
            )

            result = await poll_task_status(
                self.client,
                self.current_task_id,
                logger=self.logger,
                poll_interval=2.0,
                cancel_flag_getter=lambda: self.cancel_flag,
            )

            video_url = None
            if hasattr(result, 'content') and hasattr(result.content, 'video_url'):
                video_url = result.content.video_url
            elif hasattr(result, 'video_url'):
                video_url = result.video_url

            if not video_url:
                self.logger.error(f"Could not extract video URL from result: {result}")
                raise RuntimeError("Failed to get video URL from response")

            video_path = download_video(video_url, self.logger)
            output_meta = self._build_output_meta(input_data, result, mode, video_path)

            self.logger.info(f"Video generated successfully: {video_path}")

            return self.OutputType(video=File(path=video_path), output_meta=output_meta)

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
        finally:
            self.current_task_id = None


class SeedanceStudioApp(SeedanceApp):
    """Seedance 2.0 with the BytePlus private asset library.

    Every reference — image, video, and audio — is uploaded to a private asset
    group and passed as an asset:// URI. This is what makes the input a trusted
    asset; raw URLs trip the real-person / privacy input filters.

    Studio apps are also the only ones that may expose safety_filter and route
    to a custom unfiltered endpoint.
    """

    is_studio: ClassVar[bool] = True
    # Endpoint used when safety_filter is False. None means this app has no
    # unfiltered endpoint yet and always uses model_id.
    unfiltered_model_id: ClassVar[Optional[str]] = None

    def _select_model(self, input_data) -> str:
        """Pick the filtered or unfiltered endpoint for this request."""
        if getattr(input_data, "safety_filter", True):
            return self.model_id
        return self.unfiltered_model_id or self.model_id

    async def setup(self, metadata):
        await super().setup(metadata)
        self.asset_client = setup_asset_client()
        self.asset_group_id = None

    async def _ensure_asset_group(self, safety_identifier: Optional[str] = None) -> str:
        """Create asset group, namespaced by safety_identifier if provided."""
        if self.asset_group_id is None:
            group_name = f"seedance-studio-{safety_identifier}" if safety_identifier else "seedance-studio-assets"
            self.asset_group_id = create_asset_group(
                self.asset_client,
                name=group_name,
                description=f"Auto-managed asset group for {self.display_name}",
                logger=self.logger,
            )
        return self.asset_group_id

    async def _prepare_generation(self, input_data) -> None:
        # Ensure asset group is namespaced by safety_identifier
        await self._ensure_asset_group(input_data.safety_identifier)

    async def _resolve_uri(self, file: File, asset_type: str, label: str) -> str:
        """Upload the file to the asset library and return its asset:// URI."""
        if not file or not file.exists():
            raise RuntimeError(f"{label} does not exist: {getattr(file, 'path', file)}")
        group_id = await self._ensure_asset_group()
        return await upload_and_activate(
            self.asset_client,
            group_id,
            file.uri,
            asset_type=asset_type,
            logger=self.logger,
        )
