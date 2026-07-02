"""
Topaz Labs Denoise — video denoising.

Nyx family models for noise, compression, and artifact removal.
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
    "nyx": "nyx-3",
    "nyx-high-fidelity": "nxhf-1",
    "nyx-xl": "nxl-1",
    "nyx-fast": "nxf-1",
}


class ModelEnum(str, Enum):
    nyx = "nyx"
    nyx_high_fidelity = "nyx-high-fidelity"
    nyx_xl = "nyx-xl"
    nyx_fast = "nyx-fast"


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
        default=ModelEnum.nyx,
        description="denoise model: nyx (balanced), nyx-high-fidelity (detail preservation), nyx-xl (heavy noise), nyx-fast (speed optimized)"
    )
    video_type: VideoTypeEnum = Field(
        default=VideoTypeEnum.progressive,
        description="source video type"
    )
    focus_fix: FocusFixEnum = Field(
        default=FocusFixEnum.none,
        description="focus fix level"
    )
    compression: float = Field(default=0.0, ge=-1.0, le=1.0, description="compression artifact reduction (-1 to 1)")
    details: float = Field(default=0.0, ge=-1.0, le=1.0, description="detail enhancement (-1 to 1)")
    noise: float = Field(default=0.0, ge=-1.0, le=1.0, description="noise reduction (-1 to 1)")
    halo: float = Field(default=0.0, ge=-1.0, le=1.0, description="halo reduction (-1 to 1)")
    blur: float = Field(default=0.0, ge=-1.0, le=1.0, description="blur reduction (-1 to 1)")
    grain: float = Field(default=0.0, ge=0.0, le=0.1, description="grain amount (0-0.1)")
    recover_original_detail: float = Field(default=0.0, ge=0.0, le=1.0, description="recover original detail (0-1)")


class AppOutput(BaseAppOutput):
    video: File = Field(description="denoised video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Denoise initialized")

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

            # Build filter params (same tuning params as Proteus)
            filter_params = {
                "model": api_model,
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
            }

            if input_data.grain > 0:
                filter_params.update({
                    "grain": input_data.grain,
                    "grain_sigma": 0.5,
                    "grain_size": 1.0,
                    "grain_type": "gaussian",
                })

            # No upscaling — output = input resolution
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
            self.logger.error(f"Denoise failed: {e}")
            raise RuntimeError(f"Denoise failed: {str(e)}")
