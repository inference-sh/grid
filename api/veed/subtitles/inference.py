"""
VEED Subtitles

Add professional burned-in subtitles to videos with 25+ style presets.
Supports 100+ languages with automatic transcription or custom SRT input.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ID = "veed/subtitles"


class PresetEnum(str, Enum):
    """Subtitle style preset. Dynamic presets have richer animations; basic presets are lightweight."""
    # Dynamic presets (2x pricing multiplier)
    glass = "glass"
    whisper = "whisper"
    glide2 = "glide2"
    fusion = "fusion"
    glide = "glide"
    terminal = "terminal"
    handwritten = "handwritten"
    # Basic presets (1x pricing multiplier)
    simple = "simple"
    plain = "plain"
    beans = "beans"
    corpo = "corpo"
    boo = "boo"
    shadeplay = "shadeplay"
    casper = "casper"
    capri = "capri"
    lowkey = "lowkey"
    vinta = "vinta"
    diego = "diego"
    ali = "ali"
    slay = "slay"
    kitty = "kitty"
    hustle = "hustle"
    karl = "karl"
    sprout = "sprout"
    flex = "flex"
    mint = "mint"
    rizz = "rizz"
    vegas = "vegas"


class LanguageEnum(str, Enum):
    """Source audio language. Improves transcription accuracy. Auto-detects if omitted."""
    af_ZA = "af-ZA"
    am_ET = "am-ET"
    ar_AE = "ar-AE"
    ar_EG = "ar-EG"
    ar_SA = "ar-SA"
    bg_BG = "bg-BG"
    bs_BA = "bs-BA"
    ca_ES = "ca-ES"
    cs_CZ = "cs-CZ"
    cy_GB = "cy-GB"
    da_DK = "da-DK"
    de_DE = "de-DE"
    el_GR = "el-GR"
    en_AU = "en-AU"
    en_GB = "en-GB"
    en_IN = "en-IN"
    en_US = "en-US"
    es_ES = "es-ES"
    es_MX = "es-MX"
    es_US = "es-US"
    et_EE = "et-EE"
    fa_IR = "fa-IR"
    fi_FI = "fi-FI"
    fil_PH = "fil-PH"
    fr_CA = "fr-CA"
    fr_FR = "fr-FR"
    he_IL = "he-IL"
    hi_Latn_IN = "hi-Latn-IN"
    hr_HR = "hr-HR"
    hu_HU = "hu-HU"
    hy_AM = "hy-AM"
    id_ID = "id-ID"
    is_IS = "is-IS"
    it_IT = "it-IT"
    ja_JP = "ja-JP"
    ka_GE = "ka-GE"
    kk_KZ = "kk-KZ"
    ko_KR = "ko-KR"
    lt_LT = "lt-LT"
    lv_LV = "lv-LV"
    mk_MK = "mk-MK"
    mn_MN = "mn-MN"
    ms_MY = "ms-MY"
    nb_NO = "nb-NO"
    nl_NL = "nl-NL"
    pl_PL = "pl-PL"
    pt_BR = "pt-BR"
    pt_PT = "pt-PT"
    ro_RO = "ro-RO"
    ru_RU = "ru-RU"
    sk_SK = "sk-SK"
    sl_SI = "sl-SI"
    sq_AL = "sq-AL"
    sr_RS = "sr-RS"
    sv_SE = "sv-SE"
    sw_KE = "sw-KE"
    th_TH = "th-TH"
    tr_TR = "tr-TR"
    uk_UA = "uk-UA"
    ur_PK = "ur-PK"
    uz_UZ = "uz-UZ"
    vi_VN = "vi-VN"
    zh = "zh"
    zh_HK = "zh-HK"
    zh_TW = "zh-TW"
    zu_ZA = "zu-ZA"


class PositionEnum(str, Enum):
    """Vertical position of subtitles on screen."""
    top = "top"
    center = "center"
    bottom = "bottom"


class ShadowEnum(str, Enum):
    """Shadow intensity behind subtitle text."""
    none = "none"
    min = "min"
    mid = "mid"
    max = "max"


DYNAMIC_PRESETS = {"glass", "whisper", "glide2", "fusion", "glide", "terminal", "handwritten"}


class AppInput(BaseAppInput):
    """Input schema for VEED Subtitles."""

    video: File = Field(
        description="Video to add subtitles to.",
    )
    preset: PresetEnum = Field(
        default=PresetEnum.simple,
        description="Subtitle style preset. Dynamic presets (glass, whisper, glide2, fusion, glide, terminal, handwritten) have richer animations. Basic presets are lightweight and predictable.",
    )
    language: Optional[LanguageEnum] = Field(
        default=None,
        description="Source audio language. Improves transcription accuracy. Leave empty for auto-detection.",
    )
    srt_file: Optional[File] = Field(
        default=None,
        description="Custom .srt subtitle file. When provided, transcription is skipped.",
    )
    srt_content: Optional[str] = Field(
        default=None,
        description="Raw SRT subtitle text. Alternative to srt_file. When provided, transcription is skipped.",
    )
    position: Optional[PositionEnum] = Field(
        default=None,
        description="Override vertical position of subtitles. Uses preset default if omitted.",
    )
    shadow: Optional[ShadowEnum] = Field(
        default=None,
        description="Override shadow intensity behind text. Uses preset default if omitted.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for VEED Subtitles."""

    video: File = Field(description="Rendered video with styled subtitles.")


class App(BaseApp):
    """VEED Subtitles application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("VEED Subtitles initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "video_url": input_data.video.uri,
            "preset": input_data.preset.value,
        }

        if input_data.language is not None:
            request["language"] = input_data.language.value

        if input_data.srt_file is not None:
            request["srt_file_url"] = input_data.srt_file.uri
        elif input_data.srt_content is not None:
            request["srt_content"] = input_data.srt_content

        # Build customization overrides
        customization = {}
        if input_data.position is not None:
            customization["position"] = input_data.position.value
        if input_data.shadow is not None:
            customization["shadow"] = input_data.shadow.value
        if customization:
            request["customization"] = customization

        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

            setup_fal_client()

            is_dynamic = input_data.preset.value in DYNAMIC_PRESETS
            self.logger.info(
                f"Adding subtitles with preset '{input_data.preset.value}' ({'dynamic' if is_dynamic else 'basic'})"
            )
            if input_data.language:
                self.logger.info(f"Language: {input_data.language.value}")
            else:
                self.logger.info("Language: auto-detect")
            if input_data.srt_file or input_data.srt_content:
                self.logger.info("Using custom SRT — skipping transcription")

            request_data = self._build_request(input_data)
            result = run_fal_model(MODEL_ID, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        extra={
                            "preset": input_data.preset.value,
                            "dynamic": is_dynamic,
                        },
                    )
                ]
            )

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during subtitle rendering: {e}")
            raise RuntimeError(f"Subtitle rendering failed: {str(e)}")
