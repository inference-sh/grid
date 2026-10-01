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

import hashlib
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
    ensure_asset_group,
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
    # 2.5 supports audio-only reference input; 2.0 requires at least one
    # image or video alongside audio references.
    supports_audio_only: ClassVar[bool] = False
    # 2.5 requires adaptive ratio for first-frame, first+last-frame, and
    # multimodal tasks with video input (editing/extension). Auto-coerce
    # to prevent API rejections.
    force_adaptive_ratio: ClassVar[bool] = False
    # Draft mode (2.5): `draft` renders a cheap 480p preview and returns its
    # task id; `draft_task_id` renders the final video from that draft, reusing
    # its prompt, inputs, duration, ratio, seed and audio setting. A final from
    # a draft only accepts draft_final_resolution.
    supports_draft: ClassVar[bool] = False
    draft_final_resolution: ClassVar[str] = "1080p"

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

    async def _resolve_uri(self, file: File, asset_type: str, label: str, ctx) -> str:
        """Resolve an input file to the URI handed to the generation API.

        The plain variant passes the file's own URL through. The studio variant
        overrides this to upload the file to the asset library first.
        """
        if not file or not file.exists():
            raise RuntimeError(f"{label} does not exist: {getattr(file, 'path', file)}")
        return file.uri

    async def _prepare_generation(self, input_data):
        """Build the per-request context passed to _resolve_uri.

        Whatever this returns lives on the call stack for exactly one request.
        Nothing derived from input_data may be stored on self: workers are
        reused across end users, so per-user state would leak between them.
        """
        return None

    async def _build_content(self, input_data, mode: str, ctx) -> list:
        """Build the content list for the BytePlus API."""
        content = []

        if input_data.prompt:
            content.append(build_text_content(input_data.prompt))

        if mode == "first-last-frame":
            first_uri = await self._resolve_uri(input_data.image, "Image", "First-frame image", ctx)
            last_uri = await self._resolve_uri(input_data.end_image, "Image", "Last-frame image", ctx)
            content.append(build_image_content(first_uri, role="first_frame"))
            content.append(build_image_content(last_uri, role="last_frame"))

        elif mode == "image-to-video":
            first_uri = await self._resolve_uri(input_data.image, "Image", "Input image", ctx)
            content.append(build_image_content(first_uri, role="first_frame"))

        elif mode == "multimodal-reference":
            for ref_img in input_data.reference_images:
                if ref_img.exists():
                    uri = await self._resolve_uri(ref_img, "Image", "Reference image", ctx)
                    content.append(build_image_content(uri, role="reference_image"))

            for ref_vid in input_data.reference_videos:
                if ref_vid.exists():
                    uri = await self._resolve_uri(ref_vid, "Video", "Reference video", ctx)
                    content.append(build_video_content(uri))

            if input_data.reference_audios:
                if not self.supports_audio_only:
                    has_visual = input_data.reference_images or input_data.reference_videos
                    if not has_visual:
                        raise RuntimeError("Audio reference requires at least one image or video reference.")
                for ref_aud in input_data.reference_audios:
                    if ref_aud.exists():
                        uri = await self._resolve_uri(ref_aud, "Audio", "Reference audio", ctx)
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
            "draft": bool(getattr(input_data, "draft", False)),
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

            draft = bool(getattr(input_data, "draft", False))
            draft_task_id = (getattr(input_data, "draft_task_id", None) or "").strip()
            if draft and draft_task_id:
                raise RuntimeError("Set either draft (to make a 480p preview) or draft_task_id (to render the final video from a draft), not both.")
            if draft_task_id:
                return await self._render_from_draft(input_data, draft_task_id)

            mode = self._determine_mode(input_data)
            if mode == "text-to-video" and not (input_data.prompt or "").strip():
                raise RuntimeError("A prompt is required for text-to-video generation.")
            suffix = " (studio)" if self.is_studio else ""
            self.logger.info(f"Starting {mode} generation{suffix}")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Ratio: {input_data.ratio.value}, Duration: {input_data.duration}s, Audio: {input_data.generate_audio}")

            ctx = await self._prepare_generation(input_data)

            content = await self._build_content(input_data, mode, ctx)

            ratio = input_data.ratio.value
            if self.force_adaptive_ratio and ratio != "adaptive":
                needs_adaptive = mode in ("image-to-video", "first-last-frame")
                if not needs_adaptive and mode == "multimodal-reference" and input_data.reference_videos:
                    needs_adaptive = True
                if needs_adaptive:
                    self.logger.info(f"Coercing ratio from '{ratio}' to 'adaptive' (required for {mode} on this model)")
                    ratio = "adaptive"

            resolution = input_data.resolution.value
            if draft:
                if resolution != "480p":
                    self.logger.info(f"Draft mode: rendering at 480p instead of {resolution}")
                resolution = "480p"

            api_params = {
                "resolution": resolution,
                "ratio": ratio,
                "duration": input_data.duration,
                "generate_audio": input_data.generate_audio,
                "seed": input_data.seed,
                "watermark": input_data.watermark,
            }
            if input_data.safety_identifier:
                api_params["safety_identifier"] = input_data.safety_identifier
            task_type = getattr(input_data, "task_type", None)
            if task_type and hasattr(task_type, "value") and task_type.value != "auto":
                api_params["omni_reference_task_type"] = task_type.value
            extra_body = {}
            output_format = getattr(input_data, "output_format", None)
            if output_format:
                extra_body["output_format"] = output_format
            if draft:
                # Sent in extra_body so it works on every SDK version.
                extra_body["draft"] = True
            if extra_body:
                api_params["extra_body"] = extra_body
            self.current_task_id = create_content_task(
                self.client,
                model=self._select_model(input_data),
                content=content,
                logger=self.logger,
                **api_params,
            )
            task_id = self.current_task_id

            result, video_path = await self._wait_for_video()
            output_meta = self._build_output_meta(input_data, result, mode, video_path)

            self.logger.info(f"Video generated successfully: {video_path}")

            if draft:
                self.logger.info(f"Draft task id: {task_id}")
                return self.OutputType(video=File(path=video_path), output_meta=output_meta, draft_task_id=task_id)
            return self.OutputType(video=File(path=video_path), output_meta=output_meta)

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}{self._sensitive_image_hint(e)}")
        finally:
            self.current_task_id = None

    async def _wait_for_video(self):
        """Poll the current task to completion and download its video."""
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

        return result, download_video(video_url, self.logger)

    async def _render_from_draft(self, input_data, draft_task_id: str):
        """Render the final video from a draft task.

        The draft carries the prompt, inputs, duration, ratio, seed and audio
        setting, so the request is only the draft reference plus resolution
        (and output format). Every other input is ignored.
        """
        if not self.supports_draft:
            raise RuntimeError(f"{self.display_name} does not support draft mode.")

        resolution = self.draft_final_resolution
        if input_data.resolution.value != resolution:
            self.logger.info(f"Final from draft: rendering at {resolution} instead of {input_data.resolution.value} (the only resolution a draft final supports)")
        self.logger.info(f"Rendering final video from draft {draft_task_id}")

        api_params = {"resolution": resolution}
        if input_data.safety_identifier:
            api_params["safety_identifier"] = input_data.safety_identifier
        output_format = getattr(input_data, "output_format", None)
        if output_format:
            api_params["extra_body"] = {"output_format": output_format}

        self.current_task_id = create_content_task(
            self.client,
            model=self._select_model(input_data),
            content=[{"type": "draft_task", "draft_task": {"id": draft_task_id}}],
            logger=self.logger,
            **api_params,
        )

        result, video_path = await self._wait_for_video()
        output_meta = self._build_draft_final_meta(input_data, result, video_path)

        self.logger.info(f"Final video generated from draft: {video_path}")
        return self.OutputType(video=File(path=video_path), output_meta=output_meta)

    def _build_draft_final_meta(self, input_data, result, video_path: str) -> OutputMeta:
        """Output metadata for a final rendered from a draft.

        The request has no inputs of its own (they were billed with the draft),
        and duration, ratio and audio come from the draft, so everything is read
        from the result and the output file.
        """
        probe = probe_video(video_path)
        resolution = getattr(result, 'resolution', None) or self.draft_final_resolution
        ratio = getattr(result, 'ratio', None) or "16:9"
        if ratio == 'adaptive':
            ratio = '16:9'
        usage = getattr(result, 'usage', None)
        completion_tokens = getattr(usage, 'completion_tokens', None) if usage else None
        total_tokens = getattr(usage, 'total_tokens', None) if usage else None
        generate_audio = getattr(result, 'generate_audio', None)
        self.logger.info(f"BytePlus usage — completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, mode: draft-final, probe: {probe}")

        extra = {
            "mode": "draft-final",
            "draft": False,
            "draft_task_id": input_data.draft_task_id.strip(),
            "ratio": ratio,
            "generate_audio": generate_audio if generate_audio is not None else probe.get("has_audio", True),
            "seed": getattr(result, 'seed', None),
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if self.is_studio:
            extra["studio"] = True

        return OutputMeta(
            inputs=[],
            outputs=[
                VideoMeta(
                    width=probe.get("width", 1920),
                    height=probe.get("height", 1080),
                    resolution=RESOLUTION_MAP.get(resolution, VideoResolution.VIDEO_RES1080_P),
                    seconds=float(probe.get("seconds", getattr(result, 'duration', 5) or 5)),
                    fps=probe.get("fps", 24),
                    extra=extra,
                )
            ]
        )

    def _sensitive_image_hint(self, error: Exception) -> str:
        """Explain InputImageSensitiveContentDetected.* rejections (e.g. PrivacyInformation)."""
        if "InputImageSensitiveContentDetected" not in str(error):
            return ""
        hint = (
            " Hint: the provider screens input images that may show a real person, "
            "even when they are AI-generated."
        )
        if not self.is_studio:
            hint += (
                " The Studio variants (e.g. bytedance/seedance-2-5-studio) upload "
                "references to a private asset library as trusted assets, which is "
                "meant for character references like this."
            )
        return hint


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

    # Cap on cached group entries per worker, so a long-lived worker serving
    # many end users cannot grow this without bound.
    GROUP_CACHE_MAX: ClassVar[int] = 512

    async def setup(self, metadata):
        await super().setup(metadata)
        self.asset_client = setup_asset_client()
        # Maps a hash of the group name -> group id. Keyed by hash so no raw
        # end-user identifier is held in the worker; the group id it returns is
        # only ever handed back for the same identifier that produced it.
        self._group_cache = {}

    async def _prepare_generation(self, input_data):
        """Resolve this request's asset group and return per-request context.

        Returns a dict with group_id and skip_moderation so _resolve_uri
        can use them without storing per-request state on self.
        """
        safety_identifier = input_data.safety_identifier
        group_name = f"seedance-studio-{safety_identifier}" if safety_identifier else "seedance-studio-assets"
        key = hashlib.sha256(group_name.encode()).hexdigest()[:16]

        cached = self._group_cache.get(key)
        if not cached:
            cached = ensure_asset_group(
                self.asset_client,
                name=group_name,
                description=f"Auto-managed asset group for {self.display_name}",
                logger=self.logger,
            )
            if len(self._group_cache) >= self.GROUP_CACHE_MAX:
                self._group_cache.pop(next(iter(self._group_cache)))
            self._group_cache[key] = cached

        skip_mod = not getattr(input_data, "safety_filter", True)
        return {"group_id": cached, "skip_moderation": skip_mod}

    async def _resolve_uri(self, file: File, asset_type: str, label: str, ctx) -> str:
        """Upload the file to this request's asset group, return its asset:// URI."""
        if not file or not file.exists():
            raise RuntimeError(f"{label} does not exist: {getattr(file, 'path', file)}")
        return await upload_and_activate(
            self.asset_client,
            ctx["group_id"],
            file.uri,
            asset_type=asset_type,
            skip_moderation=ctx["skip_moderation"],
            logger=self.logger,
        )
