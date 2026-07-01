"""
Topaz Labs Video Upscale — Proteus family.

Precision video upscaling and enhancement via Topaz Labs API.
Models: Proteus, Proteus Natural, Rhea, Theia, Artemis, Dione, Gaia, Iris.
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


class ModelEnum(str, Enum):
    # Proteus
    proteus = "prob-4"
    proteus_natural = "pnat-1"
    # Rhea
    rhea = "rhea-1"
    # Theia
    theia_detail = "thd-3"
    theia_fidelity = "thf-4"
    # Artemis
    artemis_hq = "ahq-12"
    artemis_mq = "alqs-2"
    artemis_lq = "alq-13"
    artemis_medium_halo = "amqs-2"
    artemis_strong_halo = "amq-13"
    artemis_aliasing = "aaa-9"
    # Dione (deinterlacing)
    dione_dv = "ddv-3"
    dione_tv = "dtv-4"
    dione_robust = "dtd-4"
    dione_dehalo = "dtvs-2"
    dione_robust_dehalo = "dtds-2"
    # Gaia (CGI/animation)
    gaia_cg = "gcg-5"
    gaia_hq = "ghq-5"
    # Iris (face recovery)
    iris_mq = "iris-2"
    iris_lq = "iris-3"


class VideoTypeEnum(str, Enum):
    progressive = "Progressive"
    interlaced = "Interlaced"
    progressive_interlaced = "ProgressiveInterlaced"


class FocusFixEnum(str, Enum):
    none = "None"
    normal = "Normal"
    strong = "Strong"


class GrainTypeEnum(str, Enum):
    silver_rich = "silver_rich"
    gaussian = "gaussian"
    grey = "grey"


class ContainerEnum(str, Enum):
    mp4 = "mp4"
    mov = "mov"
    mkv = "mkv"


class AudioCodecEnum(str, Enum):
    aac = "AAC"
    copy = "Copy"


class AppInput(BaseAppInput):
    video: File = Field(description="Input video file (max 500MB)")
    model: ModelEnum = Field(
        default=ModelEnum.proteus,
        description="Enhancement model. proteus: general upscaling, proteus_natural: natural look, artemis_hq: high quality, gaia_cg: CGI/animation, iris_mq: face recovery"
    )
    scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Upscale factor (1.0-4.0)"
    )
    video_type: VideoTypeEnum = Field(
        default=VideoTypeEnum.progressive,
        description="Source video type"
    )
    focus_fix_level: FocusFixEnum = Field(
        default=FocusFixEnum.none,
        description="Focus fix level"
    )
    compression: float = Field(default=0.0, ge=-1.0, le=1.0, description="Compression artifact reduction (-1 to 1)")
    details: float = Field(default=0.0, ge=-1.0, le=1.0, description="Detail enhancement (-1 to 1)")
    noise: float = Field(default=0.0, ge=-1.0, le=1.0, description="Noise reduction (-1 to 1)")
    halo: float = Field(default=0.0, ge=-1.0, le=1.0, description="Halo reduction (-1 to 1)")
    blur: float = Field(default=0.0, ge=-1.0, le=1.0, description="Blur reduction (-1 to 1)")
    grain: float = Field(default=0.0, ge=0.0, le=0.1, description="Grain amount (0-0.1)")
    recover_original_detail: float = Field(default=0.0, ge=0.0, le=1.0, description="Recover original detail (0-1)")
    output_container: ContainerEnum = Field(default=ContainerEnum.mp4, description="Output container format")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Enhanced video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Video Upscale initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist: {input_data.video.path}")

            # Get source video info
            info = get_video_info(input_data.video.path)
            self.logger.info(f"Source: {info['width']}x{info['height']}, {info['duration']}s, {info['frame_rate']}fps")

            src_width = info["width"] or 1920
            src_height = info["height"] or 1080
            out_width = int(src_width * input_data.scale)
            out_height = int(src_height * input_data.scale)
            frame_rate = info["frame_rate"] or 30
            frame_count = info["frame_count"] or int((info["duration"] or 10) * frame_rate)

            # Build filter params
            filter_params = {"model": input_data.model.value}
            # Only add tuning params for models that support them (not proteus_natural)
            if input_data.model != ModelEnum.proteus_natural:
                filter_params.update({
                    "video_type": input_data.video_type.value,
                    "focus_fix_level": input_data.focus_fix_level.value,
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
                    "container": input_data.output_container.value,
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
                # Topaz returns {"uploadId": "...", "urls": ["https://..."]}
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
            download_url = (
                status_data.get("downloadUrl")
                or status_data.get("download_url")
            )
            if not download_url:
                # Topaz returns {"download": {"url": "https://..."}}
                dl = status_data.get("download", {})
                if isinstance(dl, dict):
                    download_url = dl.get("url")
            if not download_url:
                raise RuntimeError(f"No download URL in status response: {status_data}")

            output_path = download_result(download_url, self.logger)

            # Get actual credits from API response
            estimates = status_data.get("estimates", {})
            cost_range = estimates.get("cost", [])
            if cost_range:
                credits = cost_range[-1]  # use upper estimate
            else:
                # Fallback: ~4 credits per 10s at 1080p
                duration_s = info["duration"] or 10
                resolution_factor = (out_width * out_height) / (1920 * 1080)
                credits = max(1, int(duration_s / 10 * 4 * resolution_factor))

            # $100 for 1400 credits = $0.07143/credit = 7.143 cents/credit
            cost_cents = credits * (100.0 / 1400.0 * 100.0 / 100.0)  # credits * 7.143 cents

            self.logger.info(f"{credits} credits = ${credits * 100.0 / 1400.0:.2f}")

            return AppOutput(
                video=File(path=output_path),
                output_meta=OutputMeta(outputs=[RawMeta(
                    cost=cost_cents,
                    extra={
                        "credits": credits,
                        "model": input_data.model.value,
                        "scale": input_data.scale,
                        "input_resolution": f"{src_width}x{src_height}",
                        "output_resolution": f"{out_width}x{out_height}",
                        "output_size": status_data.get("outputSize", ""),
                    }
                )]),
            )

        except Exception as e:
            self.logger.error(f"Video upscale failed: {e}")
            raise RuntimeError(f"Video upscale failed: {str(e)}")
