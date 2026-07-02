"""
Topaz Labs Starlight — generative video upscaling.

Models: precise-2.5, hq, mini, sharp, fast-2.
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
    "precise-2.5": "slp-2.5",
    "hq": "slhq-1",
    "mini": "slm-1",
    "sharp": "wonder-1",
    "fast-2": "slf-2",
}


class ModelEnum(str, Enum):
    precise_2_5 = "precise-2.5"
    hq = "hq"
    mini = "mini"
    sharp = "sharp"
    fast_2 = "fast-2"


class AppInput(BaseAppInput):
    video: File = Field(description="input video file")
    model: ModelEnum = Field(
        default=ModelEnum.precise_2_5,
        description="starlight model: precise-2.5 (best quality), hq (high quality), mini (lightweight), sharp (detail emphasis), fast-2 (speed optimized)"
    )
    scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="upscale factor (1.0-4.0)"
    )
    # HQ model specific params
    upscaling_factor: Optional[float] = Field(
        default=None,
        description="upscaling factor override (hq model only)"
    )
    low_sharpness: Optional[bool] = Field(
        default=None,
        description="use low sharpness mode (hq model only)"
    )
    keep_original: Optional[bool] = Field(
        default=None,
        description="keep original detail (hq model only)"
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="upscaled video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Starlight initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist: {input_data.video.path}")

            api_model = MODEL_MAP[input_data.model.value]
            info = get_video_info(input_data.video.path)
            self.logger.info(f"Source: {info['width']}x{info['height']}, {info['duration']}s, {info['frame_rate']}fps")
            self.logger.info(f"Model: {input_data.model.value} -> {api_model}, scale: {input_data.scale}x")

            src_width = info["width"] or 1920
            src_height = info["height"] or 1080
            out_width = int(src_width * input_data.scale)
            out_height = int(src_height * input_data.scale)
            frame_rate = info["frame_rate"] or 30
            frame_count = info["frame_count"] or int((info["duration"] or 10) * frame_rate)

            # Build starlight-specific filter params
            filter_params = {
                "model": api_model,
                "input_height": src_height,
                "input_width": src_width,
                "output_height": out_height,
                "output_width": out_width,
                "input_frame_rate": frame_rate,
                "input_frame_count": min(frame_count, 9000),
            }

            # HQ model extra params
            if input_data.model == ModelEnum.hq:
                if input_data.upscaling_factor is not None:
                    filter_params["upscaling_factor"] = input_data.upscaling_factor
                if input_data.low_sharpness is not None:
                    filter_params["low_sharpness"] = input_data.low_sharpness
                if input_data.keep_original is not None:
                    filter_params["keep_original"] = input_data.keep_original

            # Step 1: Create request
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
                        "model": input_data.model.value,
                        "scale": input_data.scale,
                        "input_resolution": f"{src_width}x{src_height}",
                        "output_resolution": f"{out_width}x{out_height}",
                        "duration": info["duration"] or 0,
                    }
                )]),
            )

        except Exception as e:
            self.logger.error(f"Starlight upscale failed: {e}")
            raise RuntimeError(f"Starlight upscale failed: {str(e)}")
