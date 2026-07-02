"""
Topaz Labs Frame Interpolation — slowmo and fps boost.

Models: Apollo, Apollo Fast, Chronos, Chronos Fast, Aion.
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
    "apollo": "apo-8",
    "apollo-fast": "apf-2",
    "chronos": "chr-2",
    "chronos-fast": "chf-3",
    "aion": "aion-1",
}


class ModelEnum(str, Enum):
    apollo = "apollo"
    apollo_fast = "apollo-fast"
    chronos = "chronos"
    chronos_fast = "chronos-fast"
    aion = "aion"


class AppInput(BaseAppInput):
    video: File = Field(description="input video file")
    model: ModelEnum = Field(
        default=ModelEnum.apollo,
        description="interpolation model: apollo (best quality), apollo-fast (speed), chronos (optical flow), chronos-fast (fast optical flow), aion (large motion)"
    )
    slowmo: float = Field(
        default=2.0,
        ge=1.0,
        le=16.0,
        description="slowmo factor (1-16)"
    )
    target_fps: float = Field(
        default=60.0,
        ge=15.0,
        le=240.0,
        description="target frame rate (15-240)"
    )
    remove_duplicates: bool = Field(
        default=True,
        description="remove duplicate frames before interpolation"
    )
    duplicate_threshold: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="duplicate detection threshold (0.001-0.1)"
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="interpolated video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Frame Interpolation initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist: {input_data.video.path}")

            api_model = MODEL_MAP[input_data.model.value]
            info = get_video_info(input_data.video.path)
            self.logger.info(f"Source: {info['width']}x{info['height']}, {info['duration']}s, {info['frame_rate']}fps")
            self.logger.info(f"Model: {input_data.model.value} -> {api_model}, slowmo: {input_data.slowmo}x, target_fps: {input_data.target_fps}")

            src_width = info["width"] or 1920
            src_height = info["height"] or 1080
            frame_rate = info["frame_rate"] or 30
            frame_count = info["frame_count"] or int((info["duration"] or 10) * frame_rate)

            # Build filter params for frame interpolation
            filter_params = {
                "model": api_model,
                "slowmo": input_data.slowmo,
                "fps": input_data.target_fps,
                "duplicate": input_data.remove_duplicates,
                "duplicateThreshold": input_data.duplicate_threshold,
            }

            # Output resolution = input resolution (no upscaling)
            req = create_request(
                api_key=self.api_key,
                source={
                    "resolution": {"width": src_width, "height": src_height},
                    "container": info["container"],
                    "size": info["size"],
                    "duration": info["duration"] or 10,
                    "frameRate": frame_rate,
                    "frameCount": frame_count,
                },
                output={
                    "resolution": {"width": src_width, "height": src_height},
                    "audioCodec": "AAC",
                    "audioTransfer": "Copy",
                    "frameRate": input_data.target_fps,
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
                        "model": input_data.model.value,
                        "slowmo": input_data.slowmo,
                        "target_fps": input_data.target_fps,
                        "resolution": f"{src_width}x{src_height}",
                        "duration": info["duration"] or 0,
                    }
                )]),
            )

        except Exception as e:
            self.logger.error(f"Frame interpolation failed: {e}")
            raise RuntimeError(f"Frame interpolation failed: {str(e)}")
