"""
Topaz Labs Video Utilities — deblur, colorization, HDR.

Models: Themis-2 (motion deblur), Colorization, Hyperion-2 (SDR to HDR).
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
    "themis-2": "thm-2",
    "colorization": "color-1",
    "hyperion-2": "hyp-2",
}

# Themis-2 has tuning params, the others don't
TUNING_MODELS = {"themis-2"}


class ModelEnum(str, Enum):
    themis_2 = "themis-2"
    colorization = "colorization"
    hyperion_2 = "hyperion-2"


class VideoTypeEnum(str, Enum):
    progressive = "progressive"
    interlaced = "interlaced"
    progressive_interlaced = "progressive-interlaced"

VIDEO_TYPE_MAP = {
    "progressive": "Progressive",
    "interlaced": "Interlaced",
    "progressive-interlaced": "ProgressiveInterlaced",
}


class FocusFixEnum(str, Enum):
    none = "none"
    normal = "normal"
    strong = "strong"

FOCUS_FIX_MAP = {
    "none": "None",
    "normal": "Normal",
    "strong": "Strong",
}


class AppInput(BaseAppInput):
    video: File = Field(description="input video file")
    model: ModelEnum = Field(
        default=ModelEnum.themis_2,
        description="utility model: themis-2 (motion deblur), colorization (b&w to color), hyperion-2 (sdr to hdr)"
    )
    # Tuning params (themis-2 only, ignored for other models)
    video_type: VideoTypeEnum = Field(
        default=VideoTypeEnum.progressive,
        description="source video type (themis-2 only)"
    )
    focus_fix: FocusFixEnum = Field(
        default=FocusFixEnum.none,
        description="focus fix level (themis-2 only)"
    )
    compression: float = Field(default=0.0, ge=-1.0, le=1.0, description="compression artifact reduction, -1 to 1 (themis-2 only)")
    details: float = Field(default=0.0, ge=-1.0, le=1.0, description="detail enhancement, -1 to 1 (themis-2 only)")
    noise: float = Field(default=0.0, ge=-1.0, le=1.0, description="noise reduction, -1 to 1 (themis-2 only)")
    halo: float = Field(default=0.0, ge=-1.0, le=1.0, description="halo reduction, -1 to 1 (themis-2 only)")
    blur: float = Field(default=0.0, ge=-1.0, le=1.0, description="blur reduction, -1 to 1 (themis-2 only)")
    grain: float = Field(default=0.0, ge=0.0, le=0.1, description="grain amount, 0-0.1 (themis-2 only)")
    recover_original_detail: float = Field(default=0.0, ge=0.0, le=1.0, description="recover original detail, 0-1 (themis-2 only)")


class AppOutput(BaseAppOutput):
    video: File = Field(description="processed video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Video Utilities initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist: {input_data.video.path}")

            api_model = MODEL_MAP[input_data.model.value]
            info = get_video_info(input_data.video.path)
            self.logger.info(f"Source: {info['width']}x{info['height']}, {info['duration']}s, {info['frame_rate']}fps")
            self.logger.info(f"Model: {input_data.model.value} -> {api_model}")

            src_width = info["width"] or 1920
            src_height = info["height"] or 1080
            frame_rate = info["frame_rate"] or 30
            frame_count = info["frame_count"] or int((info["duration"] or 10) * frame_rate)

            # Build filter params
            filter_params = {"model": api_model}

            # Themis-2 gets tuning params (same as Proteus)
            if input_data.model.value in TUNING_MODELS:
                filter_params.update({
                    "video_type": VIDEO_TYPE_MAP[input_data.video_type.value],
                    "focus_fix_level": FOCUS_FIX_MAP[input_data.focus_fix.value],
                    "compression": input_data.compression,
                    "details": input_data.details,
                    "noise": input_data.noise,
                    "halo": input_data.halo,
                    "blur": input_data.blur,
                    "recover_original_detail_value": input_data.recover_original_detail,
                    "auto": "Auto",
                    "field_order": "Auto",
                })
                if input_data.grain > 0:
                    filter_params.update({
                        "grain": input_data.grain,
                        "grain_sigma": 0.5,
                        "grain_size": 1.0,
                        "grain_type": "gaussian",
                    })

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
                        "resolution": f"{src_width}x{src_height}",
                        "duration": info["duration"] or 0,
                    }
                )]),
            )

        except Exception as e:
            self.logger.error(f"Video utility failed: {e}")
            raise RuntimeError(f"Video utility failed: {str(e)}")
