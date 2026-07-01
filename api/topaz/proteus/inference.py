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


# Friendly name → Topaz API model ID
MODEL_MAP = {
    # Proteus — general precision upscaling
    "proteus": "prob-4",
    "proteus-natural": "pnat-1",
    # Rhea — detail-rich upscaling with fine controls
    "rhea": "rhea-1",
    # Theia — temporal consistency
    "theia-detail": "thd-3",
    "theia-fidelity": "thf-4",
    # Artemis — legacy footage restoration
    "artemis-hq": "ahq-12",
    "artemis-mq": "alqs-2",
    "artemis-lq": "alq-13",
    "artemis-medium-halo": "amqs-2",
    "artemis-strong-halo": "amq-13",
    "artemis-aliasing": "aaa-9",
    # Dione — deinterlacing
    "dione-dv": "ddv-3",
    "dione-tv": "dtv-4",
    "dione-robust": "dtd-4",
    "dione-dehalo": "dtvs-2",
    "dione-robust-dehalo": "dtds-2",
    # Gaia — CGI and animation
    "gaia-cg": "gcg-5",
    "gaia-hq": "ghq-5",
    # Iris — face recovery
    "iris-mq": "iris-2",
    "iris-lq": "iris-3",
}


class ModelEnum(str, Enum):
    proteus = "proteus"
    proteus_natural = "proteus-natural"
    rhea = "rhea"
    theia_detail = "theia-detail"
    theia_fidelity = "theia-fidelity"
    artemis_hq = "artemis-hq"
    artemis_mq = "artemis-mq"
    artemis_lq = "artemis-lq"
    artemis_medium_halo = "artemis-medium-halo"
    artemis_strong_halo = "artemis-strong-halo"
    artemis_aliasing = "artemis-aliasing"
    dione_dv = "dione-dv"
    dione_tv = "dione-tv"
    dione_robust = "dione-robust"
    dione_dehalo = "dione-dehalo"
    dione_robust_dehalo = "dione-robust-dehalo"
    gaia_cg = "gaia-cg"
    gaia_hq = "gaia-hq"
    iris_mq = "iris-mq"
    iris_lq = "iris-lq"

# Models that accept no tuning parameters
NO_PARAMS_MODELS = {"proteus-natural"}


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


class ContainerEnum(str, Enum):
    mp4 = "mp4"
    mov = "mov"
    mkv = "mkv"


class AppInput(BaseAppInput):
    video: File = Field(description="input video file (max 500MB)")
    model: ModelEnum = Field(
        default=ModelEnum.proteus,
        description="enhancement model: proteus (general), proteus-natural (natural look), rhea (detail-rich), theia-detail/fidelity (temporal), artemis-hq/mq/lq (legacy footage), dione-dv/tv (deinterlace), gaia-cg/hq (CGI/animation), iris-mq/lq (face recovery)"
    )
    scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="upscale factor (1.0-4.0)"
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
    output_container: ContainerEnum = Field(default=ContainerEnum.mp4, description="output container format")


class AppOutput(BaseAppOutput):
    video: File = Field(description="enhanced video")


class App(BaseApp):
    async def setup(self):
        self.logger = setup_logger(__name__)
        self.api_key = get_api_key()
        self.logger.info("Topaz Proteus initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist: {input_data.video.path}")

            # Resolve friendly name to API model ID
            api_model = MODEL_MAP[input_data.model.value]

            # Get source video info
            info = get_video_info(input_data.video.path)
            self.logger.info(f"Source: {info['width']}x{info['height']}, {info['duration']}s, {info['frame_rate']}fps")
            self.logger.info(f"Model: {input_data.model.value} -> {api_model}, scale: {input_data.scale}x")

            src_width = info["width"] or 1920
            src_height = info["height"] or 1080
            out_width = int(src_width * input_data.scale)
            out_height = int(src_height * input_data.scale)
            frame_rate = info["frame_rate"] or 30
            frame_count = info["frame_count"] or int((info["duration"] or 10) * frame_rate)

            # Build filter params
            filter_params = {"model": api_model}

            if input_data.model.value not in NO_PARAMS_MODELS:
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

            # Report raw credits — no dollar conversion in app
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
                        "output_size": status_data.get("outputSize", ""),
                    }
                )]),
            )

        except Exception as e:
            self.logger.error(f"Video upscale failed: {e}")
            raise RuntimeError(f"Video upscale failed: {str(e)}")
