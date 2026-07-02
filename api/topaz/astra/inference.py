"""
Topaz Labs Astra — creative video upscaling.

AI-guided upscaling with prompt and creativity controls.
"""

from enum import Enum
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta
from pydantic import Field
from typing import Optional

from .topaz_helper import (
    get_api_key, setup_logger, get_video_info,
    create_request, accept_request, upload_to_s3,
    complete_upload, poll_status, download_result,
)


MODEL_MAP = {
    "astra-2": "ast-2",
}


class ModelEnum(str, Enum):
    astra_2 = "astra-2"


class AppInput(BaseAppInput):
    video: File = Field(description="input video file")
    creativity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="creativity level (0-1)"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="text guidance for creative upscaling (limits input to 450 frames)"
    )
    realism: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="realism level (0-1)"
    )
    sharpness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="sharpness level (0-1)"
    )
    scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="upscale factor (1.0-4.0)"
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="upscaled video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Astra initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist: {input_data.video.path}")

            api_model = MODEL_MAP["astra-2"]
            info = get_video_info(input_data.video.path)
            self.logger.info(f"Source: {info['width']}x{info['height']}, {info['duration']}s, {info['frame_rate']}fps")

            src_width = info["width"] or 1920
            src_height = info["height"] or 1080
            out_width = int(src_width * input_data.scale)
            out_height = int(src_height * input_data.scale)
            frame_rate = info["frame_rate"] or 30
            frame_count = info["frame_count"] or int((info["duration"] or 10) * frame_rate)

            # With prompt, max 450 frames; without, 9000
            max_frames = 450 if input_data.prompt else 9000
            capped_frame_count = min(frame_count, max_frames)

            filter_params = {
                "model": api_model,
                "creativity": input_data.creativity,
                "sharpness": input_data.sharpness,
                "input_height": src_height,
                "input_width": src_width,
                "output_height": out_height,
                "output_width": out_width,
                "input_frame_rate": frame_rate,
                "input_frame_count": capped_frame_count,
            }

            if input_data.prompt is not None:
                filter_params["prompt"] = input_data.prompt
            if input_data.realism is not None:
                filter_params["realism"] = input_data.realism

            # Step 1: Create request
            req = create_request(
                api_key=self.api_key,
                source={
                    "resolution": {"width": src_width, "height": src_height},
                    "container": info["container"],
                    "size": info["size"],
                    "duration": info["duration"] or 10,
                    "frameRate": frame_rate,
                    "frameCount": capped_frame_count,
                },
                output={
                    "resolution": {"width": out_width, "height": out_height},
                    "audioCodec": "AAC",
                    "audioTransfer": "Copy",
                    "frameRate": frame_rate,
                    "dynamicCompressionLevel": "High",
                    "container": "mp4",
                },
                filters=[filter_params],
                logger=self.logger,
            )

            request_id = req.get("id") or req.get("requestId")
            if not request_id:
                raise RuntimeError(f"No request ID in response: {req}")

            # Step 2: Accept and get upload URL
            accept_data = accept_request(self.api_key, request_id, self.logger)
            upload_url = accept_data.get("uploadUrl") or accept_data.get("upload_url")
            if not upload_url:
                urls = accept_data.get("urls", [])
                if urls:
                    upload_url = urls[0]
            if not upload_url:
                raise RuntimeError(f"No upload URL in accept response: {accept_data}")

            # Step 3: Upload to S3
            etag = upload_to_s3(upload_url, input_data.video.path, logger=self.logger)

            # Step 4: Complete upload
            complete_upload(self.api_key, request_id, etag, self.logger)

            # Step 5: Poll for completion
            status_data = poll_status(self.api_key, request_id, logger=self.logger)

            # Step 6: Download result
            download_url = status_data.get("downloadUrl") or status_data.get("download_url")
            if not download_url:
                dl = status_data.get("download", {})
                if isinstance(dl, dict):
                    download_url = dl.get("url")
            if not download_url:
                raise RuntimeError(f"No download URL in status response: {status_data}")

            output_path = download_result(download_url, self.logger)

            # Report raw credits
            estimates = status_data.get("estimates", {})
            cost_range = estimates.get("cost", [])
            credits = cost_range[-1] if cost_range else 0

            self.logger.info(f"{credits} credits used")

            return AppOutput(
                video=File(path=output_path),
                output_meta=OutputMeta(outputs=[RawMeta(
                    cost=credits,
                    extra={
                        "credits": credits,
                        "model": "astra-2",
                        "creativity": input_data.creativity,
                        "scale": input_data.scale,
                        "has_prompt": input_data.prompt is not None,
                        "input_resolution": f"{src_width}x{src_height}",
                        "output_resolution": f"{out_width}x{out_height}",
                        "duration": info["duration"] or 0,
                    }
                )]),
            )

        except Exception as e:
            self.logger.error(f"Astra upscale failed: {e}")
            raise RuntimeError(f"Astra upscale failed: {str(e)}")
