"""
P-Video-Edit - Edit videos from text prompts by Pruna

Takes a source video (up to 15s) and a text prompt describing the edit.
Optional reference images (up to 4) guide identity or style.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional, List
import logging

from .pruna_helper import run_prediction, get_generation_url, download_video, upload_file


class AppInput(BaseAppInput):
    video: File = Field(
        description="Source video to edit (.mp4). Maximum length: 15 seconds."
    )
    prompt: str = Field(
        description="Text prompt describing the edit, e.g. 'Replace the sky with a sunset'."
    )
    images: Optional[List[File]] = Field(
        default=None,
        description="Optional reference image(s) for identity or style guidance (1-4 images, jpg/png/webp).",
        max_length=4,
    )
    prompt_upsampling: bool = Field(
        default=True,
        description="Enhance the edit prompt with an LLM."
    )
    draft: bool = Field(
        default=False,
        description="Draft mode: faster, lower-quality preview."
    )
    save_audio: bool = Field(
        default=True,
        description="Preserve audio from the source video."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Edited video.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):

    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-edit"
        self.logger.info("P-Video-Edit initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Editing video: {input_data.prompt[:100]}")

        # Upload source video
        if input_data.video.uri and input_data.video.uri.startswith("http"):
            video_url = input_data.video.uri
        else:
            self.logger.info("Uploading source video...")
            upload_result = upload_file(input_data.video.path, logger=self.logger)
            video_url = upload_result.get("urls", {}).get("get")
            if not video_url:
                raise RuntimeError("Failed to get URL for uploaded video")

        request_data = {
            "video": video_url,
            "prompt": input_data.prompt,
            "prompt_upsampling": input_data.prompt_upsampling,
            "draft": input_data.draft,
            "save_audio": input_data.save_audio,
        }

        # Upload reference images if provided
        if input_data.images:
            image_urls = []
            for i, img in enumerate(input_data.images):
                if img.uri and img.uri.startswith("http"):
                    image_urls.append(img.uri)
                else:
                    self.logger.info(f"Uploading reference image {i+1}...")
                    upload_result = upload_file(img.path, logger=self.logger)
                    img_url = upload_result.get("urls", {}).get("get")
                    if not img_url:
                        raise RuntimeError(f"Failed to get URL for reference image {i+1}")
                    image_urls.append(img_url)
            request_data["images"] = image_urls

        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        result = await run_prediction(
            model=self.model,
            input_data=request_data,
            use_sync=False,
            logger=self.logger,
        )

        generation_url = get_generation_url(result)
        video_path = download_video(generation_url, logger=self.logger)

        # Probe output video duration
        video_seconds = float(result.get("duration", 0))
        if video_seconds == 0:
            import subprocess
            import json as _json
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
                    capture_output=True, text=True, timeout=30,
                )
                if probe.returncode == 0:
                    fmt = _json.loads(probe.stdout).get("format", {})
                    video_seconds = float(fmt.get("duration", 0))
            except Exception:
                pass

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    seconds=video_seconds,
                    extra={"draft_mode": input_data.draft},
                )
            ],
        )

        self.logger.info(f"Video edited: {video_seconds:.1f}s, draft={input_data.draft}")

        return AppOutput(
            video=File(path=video_path),
            seed=result.get("seed"),
            output_meta=output_meta,
        )
